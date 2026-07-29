# model_build_vgg16_two_file.py
# Local, non-Docker VGGNet-16 QONNX export + FINN build script.
# This file uses only one companion file: custom_steps.py

import argparse
import os
import shutil
import sys
from pathlib import Path

import onnx
import torch
import brevitas.nn as qnn
from brevitas.export import export_qonnx

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,

    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

from brainsmith.registry import has_step
from brainsmith.dse.api import explore_design_space

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import custom_steps # noqa: F401

def assert_custom_steps_registered():
    required_steps = [
        "pre-input clean",
        "post_qonnx_to_finn",
        "pre-streamline",
        "post-streamline",
        "post_convert_to_hw_check",
        "convert_optional_hw_layers",
        "post_dataflow_partition_check",
        "vgg16_streamline",
        "vgg16_clean_transposes_before_hw",
        "vgg16_infer_hw_layers",
        "vgg16_specialize_remaining_hw_layers",
    ]

    print("\nChecking custom Brainsmith step registration:")

    for step_name in required_steps:
        ok = has_step(step_name)
        print(f"  {step_name}: {ok}")

        if not ok:
            raise RuntimeError(
                f"Custom step '{step_name}' was not registered. "
                "Make sure custom_steps.py is importable and imported "
                "before Brainsmith parses the blueprint."
            )

class QuantVGG16(torch.nn.Module):
    """VGGNet-16 / VGG16-style quantized CNN for 3x224x224 inputs."""

    def __init__(self, num_classes: int = 1000, bit_width: int = 8):
        super().__init__()

        self.quant_in = qnn.QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

        self.features = torch.nn.Sequential(
            # Block 1: 224 -> 112
            qnn.QuantConv2d(3, 64, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantConv2d(64, 64, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112 -> 56
            qnn.QuantConv2d(64, 128, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantConv2d(128, 128, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56 -> 28
            qnn.QuantConv2d(128, 256, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantConv2d(256, 256, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantConv2d(256, 256, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28 -> 14
            # Important correction compared with the uploaded script: the first
            # Block 4 convolution must consume 256 channels, not 512.
            qnn.QuantConv2d(256, 512, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantConv2d(512, 512, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantConv2d(512, 512, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 14 -> 7
            qnn.QuantConv2d(512, 512, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantConv2d(512, 512, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantConv2d(512, 512, 3, padding=1, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Five stride-2 pools convert 224x224 -> 7x7, so AdaptiveAvgPool2d is
        # not needed in the exported ONNX graph.
        self.classifier = torch.nn.Sequential(
            qnn.QuantLinear(512 * 7 * 7, 4096, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantLinear(4096, 4096, bias=True, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantLinear(4096, num_classes, bias=True, weight_bit_width=bit_width),
        )

    def forward(self, x):
        x = self.quant_in(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def export_and_clean_qonnx(raw_onnx: Path, clean_onnx: Path, num_classes: int, bit_width: int):
    """Export Brevitas QONNX, clean it while preserving quant nodes, and validate."""
    raw_onnx.parent.mkdir(parents=True, exist_ok=True)
    clean_onnx.parent.mkdir(parents=True, exist_ok=True)

    model = QuantVGG16(num_classes=num_classes, bit_width=bit_width)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    export_qonnx(
        model,
        dummy_input,
        export_path=str(raw_onnx),
        opset_version=9,
    )

    mw = ModelWrapper(str(raw_onnx))
    mw = cleanup_model(mw, preserve_qnt_ops=True)
    mw = mw.transform(InferShapes())
    mw = mw.transform(FoldConstants())
    mw = mw.transform(GiveUniqueParameterTensors())
    mw = mw.transform(GiveUniqueNodeNames())
    mw = mw.transform(GiveReadableTensorNames())
    mw = mw.transform(InferDataTypes())
    mw = mw.transform(RemoveUnusedTensors())
    mw.save(str(clean_onnx))

    onnx_model = onnx.load(str(clean_onnx))
    onnx.checker.check_model(onnx_model)
    print(f"Saved valid cleaned QONNX model: {clean_onnx}")


def run_brainsmith_dse(model_path, args):
    blueprint_path = Path(args.blueprint).resolve()
    output_dir = Path(args.output_dir).resolve()

    start_step = args.start_step
    stop_step = args.stop_step

    print("\nRunning Brainsmith DSE / FINN build:")
    print("  Model     :", Path(model_path).resolve())
    print("  Blueprint :", blueprint_path)
    print("  Output    :", output_dir)
    print("  Start-Step:", start_step if start_step is not None else "<from beginning>")
    print("  Stop-Step :", stop_step if stop_step is not None else "<to end>")
    print("  Verbosity :", args.verbosity)

    explore_design_space(
        model_path=str(Path(model_path).resolve()),
        blueprint_path=str(blueprint_path),
        output_dir=str(output_dir),
        start_step_override=start_step,
        stop_step_override=stop_step,
        verbosity=args.verbosity,
    )

def main():
    parser = argparse.ArgumentParser(
        description="VGG16 QONNX export"
    )    
    parser.add_argument(
        "--blueprint",
        default="vggnet.yaml",
        help="Blueprint YAML file",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory name or path",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output directory before building",
    )

    parser.add_argument("--raw-onnx", default="./onnx/vgg16_raw.onnx")
    parser.add_argument("--clean-onnx", default="./onnx/vgg16_clean.onnx")
    #parser.add_argument("--output-dir", default="./output/vgg16_finn_build")
    #parser.add_argument("--blueprint", default="vgg16.yaml")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--bit-width", type=int, default=8)
    #parser.add_argument("--skip-export", action="store_true")
    #parser.add_argument("--export-only", action="store_true")
    #parser.add_argument("--force", action="store_true")

    #parser.add_argument("--skip-export", action="store_true", help="Use an existing clean ONNX file.")
    #parser.add_argument("--export-only", action="store_true", help="Stop after exporting/cleaning ONNX.")
    parser.add_argument(
        "--start-step",
        default=None,
        help=(
            "Optional BrainSmith build step name to start from. "
            "Use the exact step name from vggnet.yaml, e.g. "
            "'vgg16_specialize_remaining_hw_layers'."
        ),
    )

    parser.add_argument(
        "--stop-step",
        default=None,
        help=(
            "Optional BrainSmith build step name to stop at. "
            "Use the exact step name from vggnet.yaml, e.g. "
            "'vgg16_specialize_remaining_hw_layers'."
        ),
    )

    parser.add_argument(
        "--verbosity",
        default="normal",
        choices=["quiet", "normal", "verbose", "debug"],
        help="BrainSmith logging verbosity.",
    )

    args = parser.parse_args()

    args.output_dir = os.path.abspath(args.output)

    if args.force and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    raw_onnx = Path(args.raw_onnx) if args.raw_onnx is not None else Path(args.output_dir) / "vgg16_raw.onnx"
    clean_onnx = Path(args.clean_onnx) if args.clean_onnx is not None else Path(args.output_dir) / "vgg16_clean.onnx"

    # Custom steps must be registered before Brainsmith parses YAML.
    assert_custom_steps_registered()

    print("\nOutput:", args.output_dir)
    print("Blueprint:", args.blueprint)

    #if not args.skip_export:
    export_and_clean_qonnx(
            raw_onnx=raw_onnx,
            clean_onnx=clean_onnx,
            num_classes=args.num_classes,
            bit_width=args.bit_width,
    )

    #if args.export_only:
    #    return
    
    print("\nStep 2: Run Brainsmith DSE / FINN build")
    run_brainsmith_dse(clean_onnx, args)
    print("\nBUILD COMPLETED")

if __name__ == "__main__":
    main()

