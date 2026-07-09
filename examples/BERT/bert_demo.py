#!/usr/bin/env python3
############################################################################
# BERT Demo - Brainsmith Dataflow Core
# FX-friendly static-shape BERT implementation for current torch.fx / Transformers / Brevitas.
############################################################################

import argparse
import json
import math
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
import torch.nn.functional as F
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
from transformers import BertConfig

from brainsmith.settings import get_config

# Allow this script to be run from examples/bert while still importing the local checkout.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from brainsmith import explore_design_space  # noqa: E402
from brainsmith.dse.types import SegmentStatus  # noqa: E402

warnings.simplefilter("ignore")


class FXFriendlyBertEmbeddings(nn.Module):
    """BERT embedding block with fixed static position/token-type IDs."""

    def __init__(self, config: BertConfig, batch_size: int, seq_len: int):
        super().__init__()
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.word_embeddings(input_ids)
        x = x + self.position_embeddings(self.fixed_position_ids)
        x = x + self.token_type_embeddings(self.fixed_token_type_ids)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x


class FXFriendlyBertSelfAttention(nn.Module):
    """BERT self-attention written without symbolic shape iteration.

    HuggingFace BertSelfAttention currently builds shapes from input tensors using
    tuple unpacking / star-expansion. During torch.fx tracing those dimensions are
    Proxy objects, which cannot be iterated. This implementation keeps the same
    BERT operations but uses the known quicktest batch/sequence/head dimensions.
    """

    def __init__(self, config: BertConfig, batch_size: int, seq_len: int):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.all_head_size = int(config.hidden_size)

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # Static dimensions prevent torch.fx Proxy iteration failures.
        x = x.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query_layer = self._shape(self.query(hidden_states))
        key_layer = self._shape(self.key(hidden_states))
        value_layer = self._shape(self.value(hidden_states))

        # Keep the SDPA operation in the graph. Brevitas' LLM helper can replace
        # this with qnn.ScaledDotProductAttention, after which layerwise_quantize
        # maps it to qnn.QuantScaledDotProductAttention.
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(self.batch_size, self.seq_len, self.all_head_size)
        return context_layer


class FXFriendlyBertLayer(nn.Module):
    """One BERT encoder block: MHSA + FFN with residual LayerNorms."""

    def __init__(self, config: BertConfig, batch_size: int, seq_len: int):
        super().__init__()
        self.self_attention = FXFriendlyBertSelfAttention(config, batch_size, seq_len)
        self.attention_output_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_output_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act = nn.ReLU()
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_context = self.self_attention(hidden_states)
        attention_output = self.attention_output_dense(attention_context)
        attention_output = self.attention_output_dropout(attention_output)
        hidden_states = self.attention_output_layernorm(hidden_states + attention_output)

        intermediate_output = self.intermediate_dense(hidden_states)
        intermediate_output = self.intermediate_act(intermediate_output)
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        hidden_states = self.output_layernorm(hidden_states + layer_output)
        return hidden_states


class FXFriendlyBertModel(nn.Module):
    """Small BERT model with a HuggingFace-compatible configuration interface.

    This is intentionally not a subclass of HuggingFace BertModel. It preserves the
    BERT architecture and the original demo's quantization targets, but avoids the
    dynamic Python shape/mask code that current torch.fx cannot trace.
    """

    def __init__(self, config: BertConfig, batch_size: int, seq_len: int):
        super().__init__()
        self.config = config
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)

        self.embeddings = FXFriendlyBertEmbeddings(config, batch_size, seq_len)
        self.encoder_layers = nn.ModuleList(
            [FXFriendlyBertLayer(config, batch_size, seq_len) for _ in range(config.num_hidden_layers)]
        )
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embeddings(input_ids)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        first_token = hidden_states[:, 0]
        pooled_output = self.pooler_dense(first_token)
        pooled_output = self.pooler_activation(pooled_output)
        return hidden_states, pooled_output


class FXFriendlyBertUnrolledModel(nn.Module):
    """Same as FXFriendlyBertModel, but avoids iterating over ModuleList during FX trace.

    Python iteration over ModuleList is normally traceable when the loop count is
    fixed, but using explicit getattr keeps the graph robust across torch.fx versions.
    """

    def __init__(self, config: BertConfig, batch_size: int, seq_len: int):
        super().__init__()
        self.config = config
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.num_hidden_layers = int(config.num_hidden_layers)

        self.embeddings = FXFriendlyBertEmbeddings(config, batch_size, seq_len)
        for idx in range(self.num_hidden_layers):
            setattr(self, f"encoder_layer_{idx}", FXFriendlyBertLayer(config, batch_size, seq_len))
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embeddings(input_ids)
        for idx in range(self.num_hidden_layers):
            layer = getattr(self, f"encoder_layer_{idx}")
            hidden_states = layer(hidden_states)
        first_token = hidden_states[:, 0]
        pooled_output = self.pooler_dense(first_token)
        pooled_output = self.pooler_activation(pooled_output)
        return hidden_states, pooled_output


def _make_bert_config(args: argparse.Namespace) -> BertConfig:
    """Create the same reduced BERT configuration used by the original demo."""
    return BertConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_act="relu",
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
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
    batch_size = 1

    model = FXFriendlyBertUnrolledModel(config, batch_size=batch_size, seq_len=args.seqlen)
    model.to(dtype=dtype)
    model.eval()

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, args.seqlen),
        dtype=torch.int64,
    )
    calibration_input = {"input_ids": input_ids}

    # Current torch.fx API: second positional arg is concrete_args, not input_names.
    # This model contains no dynamic shape slicing, no mask preprocessing, and no
    # Proxy iteration, so standard symbolic_trace is sufficient.
    traced = symbolic_trace(model)

    # Replace call_function scaled_dot_product_attention with Brevitas' quantizable
    # ScaledDotProductAttention module, matching the original BERT demo's intent.
    traced = replace_sdpa_with_quantizable_layers(traced)
    traced.eval()

    quant_model = layerwise_quantize(
        traced,
        compute_layer_map=_make_quantization_map(config, args.bitwidth),
    )
    quant_model.to(dtype=dtype)
    quant_model.eval()

    with torch.no_grad(), calibration_mode(quant_model):
        quant_model(**calibration_input)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with torch.no_grad():
            # Do not pass output_names here. The local remove_tail step expects
            # Brevitas' default names such as global_out_1.
            bo.export_qonnx(
                quant_model,
                (input_ids,),
                tmp_path,
                do_constant_folding=True,
                input_names=["input_ids"],
                opset_version=args.opset,
            )

        model_onnx = onnx.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    debug_path = Path(args.output_dir) / "debug_models"
    debug_path.mkdir(parents=True, exist_ok=True)
    onnx.save(model_onnx, debug_path / "00_initial_brevitas.onnx")

    print(f"  - Model inputs: {len(model_onnx.graph.input)} tensors")
    print(f"  - Model outputs: {len(model_onnx.graph.output)} tensors")
    print(f"  - Number of nodes: {len(model_onnx.graph.node)}")

    return model_onnx


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
        description="FX-friendly static-shape BERT FINN/Brainsmith demo"
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

