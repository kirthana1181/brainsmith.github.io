#!/usr/bin/env python3
############################################################################
# BERT Demo - Brainsmith Dataflow Core
# Modernized wrapper-based version for current torch.fx / transformers / Brevitas use.
############################################################################

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Tuple

import brevitas.nn as qnn
import brevitas.onnx as bo
import onnx
import torch
from torch import nn
from torch.fx import symbolic_trace

# Import local custom steps so names used in the YAML blueprint are registered.
# Keep this import even if it looks unused.
import custom_steps  # noqa: F401

from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from onnxsim import simplify
from qonnx.util.cleanup import cleanup
from transformers import BertConfig, BertModel

from brainsmith.settings import get_config

# Allow this script to be run from examples/bert while still importing the local checkout.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from brainsmith import explore_design_space  # noqa: E402
from brainsmith.dse.types import SegmentStatus  # noqa: E402

warnings.simplefilter("ignore")


class BertTraceWrapper(nn.Module):
    """Trace/export wrapper around HuggingFace BertModel.

    Current torch.fx traces by running the Python forward with Proxy tensors.
    HuggingFace BertEmbeddings normally computes position_ids/token_type_ids by
    slicing buffers with seq_length derived from input_ids.size(). During FX tracing
    that seq_length is a Proxy, so Python slicing fails.

    This wrapper keeps the public model input as input_ids, but makes all BERT
    auxiliary inputs that depend only on the fixed quicktest sequence length into
    registered buffers. That avoids proxy-valued slice bounds while preserving a
    normal ONNX/QONNX input named input_ids.
    """

    def __init__(self, bert: BertModel, batch_size: int, seq_len: int):
        super().__init__()
        self.bert = bert
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)

        # BERT expects LongTensor position/token-type IDs and an attention mask.
        # Registering them as buffers keeps them with the model and lets FX treat
        # them as fixed get_attr constants instead of dynamic Python slices.
        self.register_buffer(
            "fixed_position_ids",
            torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0).expand(self.batch_size, -1),
            persistent=False,
        )
        self.register_buffer(
            "fixed_token_type_ids",
            torch.zeros((self.batch_size, self.seq_len), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "fixed_attention_mask",
            torch.ones((self.batch_size, self.seq_len), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=self.fixed_attention_mask,
            token_type_ids=self.fixed_token_type_ids,
            position_ids=self.fixed_position_ids,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        return outputs[0], outputs[1]


def _make_bert_config(args: argparse.Namespace) -> BertConfig:
    """Create the same reduced BERT configuration used by quicktest.sh."""
    return BertConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_act="relu",
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
        # Current Transformers accepts this keyword through PretrainedConfig kwargs.
        # It keeps the model aligned with the SDPA replacement/quantization path.
        attn_implementation="sdpa",
    )


def _make_quantization_map(config: BertConfig, bitwidth: int):
    """Define the same Brevitas quantization policy as the original BERT demo."""
    unsigned_hidden_act = config.hidden_act == "relu"

    return {
        nn.Linear: (
            qnn.QuantLinear,
            {
                "input_quant": lambda module: Uint8ActPerTensorFloat
                if module.in_features == config.intermediate_size and unsigned_hidden_act
                else Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "weight_bit_width": bitwidth,
                "output_quant": None,
                "bias_quant": None,
                "return_quant_tensor": False,
            },
        ),
        qnn.ScaledDotProductAttention: (
            qnn.QuantScaledDotProductAttention,
            {
                "softmax_input_quant": Int8ActPerTensorFloat,
                "softmax_input_bit_width": bitwidth,
                "attn_output_weights_quant": Uint8ActPerTensorFloat,
                "attn_output_weights_bit_width": bitwidth,
                "q_scaled_quant": Int8ActPerTensorFloat,
                "q_scaled_bit_width": bitwidth,
                "k_transposed_quant": Int8ActPerTensorFloat,
                "k_transposed_bit_width": bitwidth,
                "v_quant": Int8ActPerTensorFloat,
                "v_bit_width": bitwidth,
                "attn_output_quant": None,
                "return_quant_tensor": False,
            },
        ),
        nn.Tanh: (
            qnn.QuantTanh,
            {
                "input_quant": None,
                "act_quant": Int8ActPerTensorFloat,
                "act_bit_width": bitwidth,
                "return_quant_tensor": False,
            },
        ),
    }


def generate_bert_model(args: argparse.Namespace) -> onnx.ModelProto:
    """Generate the quantized QONNX BERT model used as Brainsmith input."""
    dtype = torch.float32
    torch.manual_seed(args.seed)

    config = _make_bert_config(args)

    bert = BertModel(config=config)
    bert.to(dtype=dtype)
    bert.eval()

    batch_size = 1
    wrapped = BertTraceWrapper(bert, batch_size=batch_size, seq_len=args.seqlen)
    wrapped.to(dtype=dtype)
    wrapped.eval()

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, args.seqlen),
        dtype=torch.int64,
    )
    calibration_input = {"input_ids": input_ids}

    # Current torch.fx API: second positional arg is concrete_args, not input_names.
    # Therefore pass only the wrapper module here.
    traced = symbolic_trace(wrapped)

    # Replace SDPA regions with Brevitas quantizable attention layers.
    traced = replace_sdpa_with_quantizable_layers(traced)
    traced.eval()

    quant_model = layerwise_quantize(
        traced,
        compute_layer_map=_make_quantization_map(config, args.bitwidth),
    )
    quant_model.to(dtype=dtype)
    quant_model.eval()

    # One deterministic calibration pass, matching the quicktest single-input flow.
    with torch.no_grad(), calibration_mode(quant_model):
        quant_model(**calibration_input)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with torch.no_grad():
            bo.export_qonnx(
                quant_model,
                (input_ids,),
                tmp_path,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["last_hidden_state", "pooler_output"],
                opset_version=args.opset,
            )

        model = onnx.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    debug_path = Path(args.output_dir) / "debug_models"
    debug_path.mkdir(parents=True, exist_ok=True)
    onnx.save(model, debug_path / "00_initial_brevitas.onnx")

    print(f"  - Model inputs: {len(model.graph.input)} tensors")
    print(f"  - Model outputs: {len(model.graph.output)} tensors")
    print(f"  - Number of nodes: {len(model.graph.node)}")

    return model


def run_brainsmith_dse(model: onnx.ModelProto, args: argparse.Namespace):
    """Simplify/clean the QONNX model and launch the Brainsmith blueprint flow."""
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "intermediate_models"
    debug_dir = output_dir / "debug_models"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    model, check = simplify(model)
    if not check:
        raise RuntimeError("onnxsim could not simplify the exported Brevitas BERT model")

    simp_path = model_dir / "simp.onnx"
    onnx.save(model, simp_path)
    onnx.save(model, debug_dir / "01_after_simplify.onnx")

    df_input_path = output_dir / "df_input.onnx"
    cleanup(in_file=str(simp_path), out_file=str(df_input_path))
    shutil.copy2(df_input_path, debug_dir / "02_after_qonnx_cleanup.onnx")

    blueprint_path = Path(__file__).resolve().parent / args.blueprint
    if not blueprint_path.exists():
        raise FileNotFoundError(
            f"Blueprint not found: {blueprint_path}. "
            "Pass --blueprint bert_demo.yaml or keep bert_quicktest.yaml next to this script."
        )

    results = explore_design_space(
        model_path=str(df_input_path),
        blueprint_path=str(blueprint_path),
        output_dir=str(output_dir),
    )

    stats = results.compute_stats()
    if stats.get("successful", 0) == 0:
        raise RuntimeError("No successful Brainsmith builds")

    final_model_dst = output_dir / "output.onnx"
    for _, result in results.segment_results.items():
        if result.status == SegmentStatus.COMPLETED and result.output_model:
            shutil.copy2(result.output_model, final_model_dst)
            break

    handover_file = output_dir / "stitched_ip" / "shell_handover.json"
    if handover_file.exists():
        with open(handover_file, "r", encoding="utf-8") as fp:
            handover = json.load(fp)
        handover["num_layers"] = args.num_hidden_layers
        with open(handover_file, "w", encoding="utf-8") as fp:
            json.dump(handover, fp, indent=4)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper-based BERT FINN/Brainsmith demo for current torch.fx + Transformers APIs"
    )

    parser.add_argument("-o", "--output", required=True, help="Output build directory name")
    parser.add_argument("-z", "--hidden_size", type=int, default=384, help="BERT hidden_size")
    parser.add_argument(
        "-n", "--num_attention_heads", type=int, default=12, help="BERT num_attention_heads"
    )
    parser.add_argument("-l", "--num_hidden_layers", type=int, default=1, help="BERT layers")
    parser.add_argument("-i", "--intermediate_size", type=int, default=1536, help="MLP size")
    parser.add_argument("-b", "--bitwidth", type=int, default=8, help="Quantization bit width")
    parser.add_argument("-q", "--seqlen", type=int, default=128, help="Sequence length")

    parser.add_argument(
        "--blueprint",
        type=str,
        default="bert_demo.yaml",
        help="Blueprint YAML file next to this script",
    )
    parser.add_argument("--force", action="store_true", help="Remove existing output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for input/calibration tensors")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version for QONNX export")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_dir = get_config().build_dir
    args.output_dir = os.path.join(str(build_dir), args.output)

    if args.force and os.path.exists(args.output_dir):
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    print("=" * 60)
    print("BERT Demo - Brainsmith Dataflow Core")
    print("=" * 60)
    print(
        "Model: "
        f"{args.num_hidden_layers} layers, hidden={args.hidden_size}, "
        f"heads={args.num_attention_heads}, intermediate={args.intermediate_size}"
    )
    print(f"Quantization: {args.bitwidth}-bit, sequence length={args.seqlen}")
    print(f"Blueprint: {args.blueprint}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    try:
        print("\nStep 1: Generating quantized BERT model...")
        model = generate_bert_model(args)

        print("\nStep 2: Creating dataflow core accelerator...")
        run_brainsmith_dse(model, args)

        print("\n" + "=" * 70)
        print("BUILD COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Output directory: {args.output_dir}")
    except Exception as exc:
        print(f"\nERROR: Build failed with error: {exc}")
        raise


if __name__ == "__main__":
    main()

