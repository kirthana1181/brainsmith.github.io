from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


# ---------------------------------------------------------------------
# Folding config block generators
# ---------------------------------------------------------------------

def mvau_config(
    simd: int,
    pe: int,
    runtime_writeable_weights: int = 0,
    mem_mode: str = "internal_decoupled",
    ram_style: str = "auto",
    res_type: str = "auto",
) -> dict:
    return {
        "PE": int(pe),
        "SIMD": int(simd),
        "ram_style": ram_style,
        "resType": res_type,
        "mem_mode": mem_mode,
        "runtime_writeable_weights": int(runtime_writeable_weights),
    }


def thresholding_config(pe: int, runtime_writeable_weights: int = 0) -> dict:
    return {
        "PE": int(pe),
        "runtime_writeable_weights": int(runtime_writeable_weights),
        "depth_trigger_uram": 0,
        "depth_trigger_bram": 0,
    }


def convinpgen_config(simd: int) -> dict:
    # ConvInpGen SIMD should normally match the consuming MVAU SIMD.
    return {
        "SIMD": int(simd),
    }


def labelselect_config(pe: int) -> dict:
    return {
        "PE": int(pe),
    }


def generic_pe_config(pe: int) -> dict:
    return {
        "PE": int(pe),
        "ram_style": "auto",
    }


# ---------------------------------------------------------------------
# Node identification helpers
# ---------------------------------------------------------------------

def node_text(node) -> str:
    return f"{node.name or ''} {node.op_type or ''}"


def is_mvau(node) -> bool:
    text = node_text(node)
    return "MVAU" in text or "MatrixVectorActivation" in text


def is_thresholding(node) -> bool:
    return "Thresholding" in node_text(node)


def is_convinpgen(node) -> bool:
    text = node_text(node)
    return "ConvolutionInputGenerator" in text or "ConvInpGen" in text


def is_labelselect(node) -> bool:
    return "LabelSelect" in node_text(node)


def is_channelwise_or_elementwise(node) -> bool:
    text = node_text(node)
    return any(
        key in text
        for key in [
            "ChannelwiseOp",
            "Elementwise",
            "ElementWise",
            "Elementwise_apply_op",
            "AddStreams",
            "DuplicateStreams",
        ]
    )


# ---------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------

def safe_get_custom_op(node):
    try:
        return getCustomOp(node)
    except Exception:
        return None


def get_attr(node, *names: str) -> Optional[Any]:
    inst = safe_get_custom_op(node)
    if inst is None:
        return None

    try:
        attr_types = inst.get_nodeattr_types()
    except Exception:
        attr_types = {}

    for name in names:
        if name in attr_types:
            try:
                return inst.get_nodeattr(name)
            except Exception:
                pass

    return None


def get_int_attr(node, *names: str) -> Optional[int]:
    value = get_attr(node, *names)
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def node_has_attr(node, attr_name: str) -> bool:
    inst = safe_get_custom_op(node)
    if inst is None:
        return False
    try:
        return attr_name in inst.get_nodeattr_types()
    except Exception:
        return False


# ---------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------

def build_graph_maps(model: ModelWrapper):
    producer = {}
    consumers = {}

    for node in model.graph.node:
        for out_name in node.output:
            producer[out_name] = node
        for inp_name in node.input:
            consumers.setdefault(inp_name, []).append(node)

    return producer, consumers


def find_direct_predecessors(node, producer):
    return [producer[inp] for inp in node.input if inp in producer]


def find_direct_successors(node, consumers):
    successors = []
    for out_name in node.output:
        successors.extend(consumers.get(out_name, []))
    return successors


def get_last_non_one_dim(model: ModelWrapper, tensor_name: str) -> Optional[int]:
    try:
        shape = model.get_tensor_shape(tensor_name)
    except Exception:
        return None

    if shape is None:
        return None

    dims = []
    for dim in shape:
        try:
            dim_int = int(dim)
            if dim_int != 1:
                dims.append(dim_int)
        except Exception:
            pass

    if not dims:
        return None

    return dims[-1]


def infer_mvau_dims(node) -> Tuple[Optional[int], Optional[int]]:
    # FINN MVAU commonly uses MW/MH, sometimes MatrixW/MatrixH.
    mw = get_int_attr(node, "MW", "MatrixW")
    mh = get_int_attr(node, "MH", "MatrixH")
    return mw, mh


def infer_channel_count(model: ModelWrapper, node) -> Optional[int]:
    # Try common FINN attributes first.
    ch = get_int_attr(
        node,
        "NumChannels",
        "numChannels",
        "Channels",
        "num_channels",
        "PE",
        "MH",
        "MatrixH",
    )

    if ch is not None:
        return ch

    if len(node.output) > 0:
        return get_last_non_one_dim(model, node.output[0])

    return None


def infer_convinpgen_ifm_channels(node) -> Optional[int]:
    return get_int_attr(
        node,
        "IFMChannels",
        "ifm_ch",
        "Channels",
        "NumChannels",
        "numChannels",
    )


def find_preceding_convinpgen(node, producer) -> Optional[Any]:
    # Usually direct: ConvInpGen -> MVAU. Keep a small recursive search to
    # tolerate simple DataWidthConverter-like nodes in between.
    visited = set()
    stack = find_direct_predecessors(node, producer)

    while stack:
        pred = stack.pop(0)
        if pred.name in visited:
            continue
        visited.add(pred.name)

        if is_convinpgen(pred):
            return pred

        # Only look through simple stream infrastructure/adaptor nodes.
        if any(key in node_text(pred) for key in ["DataWidthConverter", "StreamingFIFO"]):
            stack.extend(find_direct_predecessors(pred, producer))

    return None


# ---------------------------------------------------------------------
# Divisor / folding selection helpers
# ---------------------------------------------------------------------

def divisors(n: int) -> list[int]:
    if n <= 0:
        return [1]
    return [d for d in range(1, n + 1) if n % d == 0]


def choose_divisor_at_most(n: Optional[int], preferred: int) -> int:
    if n is None or n <= 0:
        return 1
    valid = [d for d in divisors(n) if d <= preferred]
    return max(valid) if valid else 1


def next_power_of_two_geq(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def choose_vgg_legal_simd(
    mw: Optional[int],
    current_simd: Optional[int],
    preferred_simd: int,
    max_mw_per_simd: int,
    ifm_channels: Optional[int],
    max_simd: Optional[int],
    prefer_power2: bool,
) -> Tuple[int, int, str]:
    """
    Choose SIMD for VGG MVAU.

    Requirements:
      1. SIMD divides MW.
      2. SIMD >= ceil(MW / max_mw_per_simd), where default max_mw_per_simd=1024.
      3. If IFMChannels is known from preceding ConvInpGen, prefer/require SIMD
         to divide IFMChannels so ConvInpGen can match the MVAU SIMD.

    For the failing node MW=4608 and IFMChannels=512:
      minimum from HLS rule = ceil(4608/1024) = 5.
      SIMD=6 divides MW, but not 512.
      SIMD=8 divides both MW and 512, so this script chooses 8.
    """
    if mw is None or mw <= 0:
        return max(1, int(preferred_simd)), 1, "missing MW; used preferred SIMD"

    min_required = int(math.ceil(float(mw) / float(max_mw_per_simd)))
    min_required = max(1, min_required)

    # Do not reduce an existing legal SIMD.
    if current_simd is not None and current_simd > 0:
        current_ok = current_simd >= min_required and mw % current_simd == 0
        if ifm_channels is not None and ifm_channels > 0:
            current_ok = current_ok and (ifm_channels % current_simd == 0)
        if max_simd is not None:
            current_ok = current_ok and current_simd <= max_simd
        if current_ok:
            return current_simd, min_required, "kept current legal SIMD"

    lower = max(min_required, int(preferred_simd))

    candidates = []
    for d in divisors(mw):
        if d < lower:
            continue
        if max_simd is not None and d > max_simd:
            continue
        if ifm_channels is not None and ifm_channels > 0 and ifm_channels % d != 0:
            continue
        candidates.append(d)

    if not candidates:
        # Relax IFMChannels constraint as a last resort, but still obey HLS rule.
        for d in divisors(mw):
            if d < lower:
                continue
            if max_simd is not None and d > max_simd:
                continue
            candidates.append(d)

    if not candidates:
        raise RuntimeError(
            f"Could not find legal SIMD for MW={mw}, min_required={min_required}, "
            f"preferred={preferred_simd}, ifm_channels={ifm_channels}, max_simd={max_simd}"
        )

    if prefer_power2:
        p = next_power_of_two_geq(lower)
        power2_candidates = [d for d in candidates if d >= p and (d & (d - 1)) == 0]
        if power2_candidates:
            return min(power2_candidates), min_required, "selected power-of-two legal SIMD"

    return min(candidates), min_required, "selected smallest legal SIMD"


def choose_vgg_legal_pe(mh: Optional[int], current_pe: Optional[int], preferred_pe: int) -> int:
    if mh is None or mh <= 0:
        return max(1, int(preferred_pe))

    if current_pe is not None and current_pe > 0 and mh % current_pe == 0:
        return int(current_pe)

    return choose_divisor_at_most(mh, preferred_pe)


def estimate_mvau_cycles(mw: Optional[int], mh: Optional[int], simd: int, pe: int) -> Optional[int]:
    if mw is None or mh is None or simd <= 0 or pe <= 0:
        return None
    if mw % simd != 0 or mh % pe != 0:
        return None
    return (mw // simd) * (mh // pe)


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------

def generate_config(args) -> tuple[dict, dict]:
    model = ModelWrapper(args.model)
    producer, consumers = build_graph_maps(model)

    config: Dict[str, Dict[str, Any]] = {
        "Defaults": {}
    }

    report = {
        "mvau": [],
        "thresholding": [],
        "convinpgen": [],
        "labelselect": [],
        "generic_pe": [],
        "skipped": [],
    }

    # First pass: configure all MVAUs.
    # This gives us MVAU SIMD/PE values that ConvInpGen and Thresholding can match.
    mvau_by_name = {}
    mvau_by_node_id = {}

    for node in model.graph.node:
        if not is_mvau(node):
            continue

        if not node.name:
            report["skipped"].append((node.name, node.op_type, "unnamed MVAU"))
            continue

        mw, mh = infer_mvau_dims(node)
        current_simd = get_int_attr(node, "SIMD")
        current_pe = get_int_attr(node, "PE")

        pred_cig = find_preceding_convinpgen(node, producer)
        ifm_ch = infer_convinpgen_ifm_channels(pred_cig) if pred_cig is not None else None

        simd, min_required, reason = choose_vgg_legal_simd(
            mw=mw,
            current_simd=current_simd,
            preferred_simd=args.simd,
            max_mw_per_simd=args.max_mw_per_simd,
            ifm_channels=ifm_ch,
            max_simd=args.max_simd,
            prefer_power2=not args.no_prefer_power2,
        )

        pe = choose_vgg_legal_pe(mh=mh, current_pe=current_pe, preferred_pe=args.pe)

        config[node.name] = mvau_config(
            simd=simd,
            pe=pe,
            runtime_writeable_weights=args.runtime_writeable_weights,
            mem_mode=args.mvau_mem_mode,
            ram_style=args.mvau_ram_style,
            res_type=args.mvau_res_type,
        )

        approx_cycles = estimate_mvau_cycles(mw, mh, simd, pe)

        info = {
            "node": node,
            "MW": mw,
            "MH": mh,
            "SIMD": simd,
            "PE": pe,
            "current_SIMD": current_simd,
            "current_PE": current_pe,
            "IFMChannels": ifm_ch,
        }
        mvau_by_name[node.name] = info
        mvau_by_node_id[id(node)] = info

        report["mvau"].append(
            {
                "name": node.name,
                "op_type": node.op_type,
                "MW": mw,
                "MH": mh,
                "IFMChannels": ifm_ch,
                "current_SIMD": current_simd,
                "SIMD": simd,
                "min_required_SIMD": min_required,
                "current_PE": current_pe,
                "PE": pe,
                "approx_cycles": approx_cycles,
                "reason": reason,
            }
        )

    # Second pass: configure the non-MVAU HW nodes around the MVAUs.
    for node in model.graph.node:
        if not node.name or is_mvau(node):
            continue

        if is_thresholding(node):
            ch = infer_channel_count(model, node)
            pe = choose_divisor_at_most(ch, args.threshold_pe)

            # If Thresholding directly follows an MVAU, use the same PE when legal.
            for pred in find_direct_predecessors(node, producer):
                if pred.name in mvau_by_name:
                    mvau_pe = mvau_by_name[pred.name]["PE"]
                    if ch is None or ch % mvau_pe == 0:
                        pe = mvau_pe

            config[node.name] = thresholding_config(
                pe=pe,
                runtime_writeable_weights=0,
            )

            report["thresholding"].append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "channels": ch,
                    "PE": pe,
                }
            )

        elif is_convinpgen(node):
            ifm_ch = infer_convinpgen_ifm_channels(node)
            simd = choose_divisor_at_most(ifm_ch, args.convinpgen_simd)

            # If ConvInpGen feeds an MVAU, match that MVAU's SIMD.
            for succ in find_direct_successors(node, consumers):
                if succ.name in mvau_by_name:
                    mvau_simd = mvau_by_name[succ.name]["SIMD"]
                    if ifm_ch is None or ifm_ch % mvau_simd == 0:
                        simd = mvau_simd

            config[node.name] = convinpgen_config(simd=simd)

            report["convinpgen"].append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "IFMChannels": ifm_ch,
                    "SIMD": simd,
                }
            )

        elif is_labelselect(node):
            pe = int(args.labelselect_pe)
            config[node.name] = labelselect_config(pe=pe)

            report["labelselect"].append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "PE": pe,
                }
            )

        elif is_channelwise_or_elementwise(node):
            ch = infer_channel_count(model, node)
            pe = choose_divisor_at_most(ch, args.other_pe)

            # Only write PE configs for nodes that actually expose PE.
            # This prevents ApplyConfig from trying to set unsupported attrs.
            if node_has_attr(node, "PE"):
                config[node.name] = generic_pe_config(pe=pe)

            report["generic_pe"].append(
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "channels": ch,
                    "PE": pe,
                    "written": node.name in config,
                }
            )

    if len(config) == 1:
        raise RuntimeError(
            "No VGG16 hardware nodes were configured. Run this on the ONNX graph "
            "after build_hw_graph / vgg16_specialize_remaining_hw_layers, for example "
            "vgg16_custom_steps/13b_vgg16_specialized_remaining_hw_layers.onnx."
        )

    return config, report


# ---------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------

def print_report(report: dict):
    print("\nConfigured MVAU nodes:")
    for item in report["mvau"]:
        flag = ""
        if item["MW"] is not None and item["SIMD"] is not None:
            if item["SIMD"] < item["min_required_SIMD"]:
                flag = "  <-- ERROR"
            elif item["current_SIMD"] != item["SIMD"]:
                flag = "  <-- changed"

        print(
            f"  {item['name']} ({item['op_type']}): "
            f"MW={item['MW']}, MH={item['MH']}, IFMChannels={item['IFMChannels']}, "
            f"old_SIMD={item['current_SIMD']}, SIMD={item['SIMD']}, "
            f"min_SIMD={item['min_required_SIMD']}, "
            f"old_PE={item['current_PE']}, PE={item['PE']}, "
            f"approx_cycles={item['approx_cycles']}, {item['reason']}{flag}"
        )

    print("\nConfigured Thresholding nodes:")
    for item in report["thresholding"]:
        print(
            f"  {item['name']} ({item['op_type']}): "
            f"channels={item['channels']}, PE={item['PE']}"
        )

    print("\nConfigured ConvolutionInputGenerator nodes:")
    for item in report["convinpgen"]:
        print(
            f"  {item['name']} ({item['op_type']}): "
            f"IFMChannels={item['IFMChannels']}, SIMD={item['SIMD']}"
        )

    print("\nConfigured LabelSelect nodes:")
    for item in report["labelselect"]:
        print(f"  {item['name']} ({item['op_type']}): PE={item['PE']}")

    print("\nConfigured other PE-based nodes:")
    for item in report["generic_pe"]:
        suffix = "" if item["written"] else " (not written; no PE attr)"
        print(
            f"  {item['name']} ({item['op_type']}): "
            f"channels={item['channels']}, PE={item['PE']}{suffix}"
        )

    if report["skipped"]:
        print("\nSkipped nodes:")
        for name, op_type, reason in report["skipped"]:
            print(f"  {name} ({op_type}): {reason}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a VGG16-specific FINN/BrainSmith folding configuration JSON."
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Post-HW/pre-codegen ONNX model, e.g. "
            "results/root/vgg16_custom_steps/13b_vgg16_specialized_remaining_hw_layers.onnx "
            "or an intermediate model after build_hw_graph/specialization."
        ),
    )

    parser.add_argument(
        "--output",
        default="configs/vgg16_folding_config_auto.json",
        help="Output folding config JSON path.",
    )

    parser.add_argument(
        "--pe",
        type=int,
        default=4,
        help="Preferred PE for MVAU nodes. Existing legal PE is preserved; otherwise choose valid divisor of MH <= this value.",
    )

    parser.add_argument(
        "--simd",
        type=int,
        default=4,
        help=(
            "Preferred lower-bound SIMD for MVAU nodes. The script will increase it when required by "
            "the VGG16/HLS rule SIMD >= ceil(MW/1024)."
        ),
    )

    parser.add_argument(
        "--max-simd",
        type=int,
        default=64,
        help="Maximum SIMD to select automatically. Increase if needed for larger MW values.",
    )

    parser.add_argument(
        "--max-mw-per-simd",
        type=int,
        default=1024,
        help="HLS MVAU legality threshold. FINN requires MW/SIMD <= this value. Default: 1024.",
    )

    parser.add_argument(
        "--no-prefer-power2",
        action="store_true",
        help="Use smallest legal divisor instead of preferring power-of-two SIMD values.",
    )

    parser.add_argument(
        "--threshold-pe",
        type=int,
        default=4,
        help="Preferred PE for Thresholding nodes.",
    )

    parser.add_argument(
        "--convinpgen-simd",
        type=int,
        default=4,
        help="Preferred SIMD for ConvolutionInputGenerator nodes when not matched to a following MVAU.",
    )

    parser.add_argument(
        "--other-pe",
        type=int,
        default=4,
        help="Preferred PE for Channelwise/Elementwise-like nodes.",
    )

    parser.add_argument(
        "--labelselect-pe",
        type=int,
        default=1,
        help="Preferred PE for LabelSelect nodes.",
    )

    parser.add_argument(
        "--runtime-writeable-weights",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether MVAU weights should be runtime writeable.",
    )

    parser.add_argument(
        "--mvau-mem-mode",
        default="internal_decoupled",
        choices=["internal_decoupled", "internal_embedded", "external"],
        help="MVAU mem_mode to write into the config.",
    )

    parser.add_argument(
        "--mvau-ram-style",
        default="auto",
        help="MVAU ram_style to write into the config.",
    )

    parser.add_argument(
        "--mvau-res-type",
        default="auto",
        help="MVAU resType to write into the config.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config, report = generate_config(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    print(f"\nWrote VGG16 folding config to: {output_path}")
    print(f"Total configured nodes: {len(config) - 1}")
    print("Top-level Defaults key included for QONNX/FINN ApplyConfig compatibility.")

    print_report(report)

    print("\nAdd this to vggnet.yaml under finn_config:")
    print(f'  folding_config_file: "{output_path.resolve()}"')


if __name__ == "__main__":
    main()

