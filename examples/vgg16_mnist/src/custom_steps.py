# custom_steps.py
# Brainsmith-registered custom FINN/QONNX build steps for Quantized VGGNet-16.
#
# This file is intended to be imported by model.py BEFORE Brainsmith parses
# vggnet.yaml. The blueprint can then reference these step names directly.

import json
import os
import logging

import onnx

from brainsmith.registry import step

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name

from finn.builder.build_dataflow_config import DataflowBuildConfig

from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
    SortGraph,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

from qonnx.transformation.remove import RemoveIdentityOps
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.infer_quant_avg_pool_2d import AvgPoolAndTruncToQuantAvgPool

from finn.transformation.streamline import Streamline
from finn.transformation.streamline.absorb import (
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoMultiThreshold,
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
)
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
    MoveAddPastConv,
    MoveScalarMulPastConv,
    MoveScalarAddPastMatMul,
    MoveScalarMulPastMatMul,
    MoveMaxPoolPastMultiThreshold,
)

from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw


from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _custom_out_dir(cfg: DataflowBuildConfig):
    """Directory for VGG16 custom-step debug ONNX files and op histograms."""
    out_dir = os.path.join(cfg.output_dir, "vgg16_custom_steps")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _has_finn_hw_custom_ops(model: ModelWrapper):
    for node in model.graph.node:
        if node.domain.startswith("finn.custom_op"):
            return True
        if node.op_type.endswith("_hls") or node.op_type.endswith("_rtl"):
            return True
    return False

def _save_and_check(model: ModelWrapper, cfg: DataflowBuildConfig, name: str):
    """Save an intermediate model, run ONNX checker, and dump an op histogram."""
    out_dir = _custom_out_dir(cfg)
    model_path = os.path.join(out_dir, f"{name}.onnx")
    model.save(model_path)

    if not _has_finn_hw_custom_ops(model):
        try:
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            raise RuntimeError(f"ONNX checker failed after {name}: {e}")
    else:
        print(
            f"[vgg16 custom step] skipped generic ONNX checker after {name} "
            f"because FINN custom HW ops are present."
        )

    op_hist = {}
    for node in model.graph.node:
        op_hist[node.op_type] = op_hist.get(node.op_type, 0) + 1

    hist_path = os.path.join(out_dir, f"{name}_op_hist.json")
    with open(hist_path, "w") as f:
        json.dump(op_hist, f, indent=2)

    print(f"[vgg16 custom step] saved: {model_path}")
    print(f"[vgg16 custom step] op histogram: {hist_path}")

    return model


def _tidy(model: ModelWrapper):
    """Safe cleanup after major graph rewrites."""
    for trn in [
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
        InferShapes(),
        InferDataTypes(),
        InferDataLayouts(),
        RemoveStaticGraphInputs(),
        RemoveUnusedTensors(),
        SortGraph(),
    ]:
        model = model.transform(trn)
    return model


def _count_ops(model: ModelWrapper, op_type: str):
    return len(model.get_nodes_by_op_type(op_type))


def _assert_no_ops(model: ModelWrapper, forbidden_ops, stage_name: str):
    """Fail early if forbidden op types remain after a lowering stage."""
    found = {}
    for op in forbidden_ops:
        count = _count_ops(model, op)
        if count > 0:
            found[op] = count

    if found:
        raise RuntimeError(
            f"Unexpected ONNX ops still present after {stage_name}: {found}"
        )


def _assert_no_adds(model: ModelWrapper, stage_name: str):
    """
    Reject ResNet-style skip/residual Add nodes.

    VGG16 should be a straight feed-forward CNN. Bias/scale Add nodes are allowed
    when at least one input is an initializer. Residual-like Adds are those where
    both inputs are dynamic activation tensors.
    """
    initializer_names = {x.name for x in model.graph.initializer}
    residual_candidates = []

    for node in model.graph.node:
        if node.op_type != "Add":
            continue

        dynamic_inputs = [
            inp for inp in node.input
            if inp != "" and inp not in initializer_names
        ]

        if len(dynamic_inputs) >= 2:
            residual_candidates.append(
                {
                    "node_name": node.name,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                }
            )

    if residual_candidates:
        raise RuntimeError(
            f"Residual/skip-like dynamic Add node(s) found after {stage_name}: "
            f"{json.dumps(residual_candidates, indent=2)}"
        )


def _print_transposes(model: ModelWrapper, tag: str):
    transposes = model.get_nodes_by_op_type("Transpose")
    print(f"\n[vgg16 {tag}] Transpose count = {len(transposes)}")

    for node in transposes:
        perm = None
        for attr in node.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)

        print(f"  name : {node.name}")
        print(f"  perm : {perm}")
        print(f"  input: {list(node.input)}")
        print(f"  out  : {list(node.output)}")

@step(name="pre-input clean")
def step_vgg16_pre_qonnx_to_finn_clean(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Cleanup before Brainsmith/FINN runs step_qonnx_to_finn.

    This step must preserve QONNX Quant/Trunc/BinaryQuant nodes because
    step_qonnx_to_finn still needs them.
    """
    for trn in [
        GiveUniqueParameterTensors(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
        InferShapes(),
        FoldConstants(),
        DoubleToSingleFloat(),
        GemmToMatMul(),
        FoldTransposeIntoQuantInit(),
        FoldQuantWeights(),
        InferDataTypes(),
        InferDataLayouts(),
        RemoveIdentityOps(),
        RemoveStaticGraphInputs(),
        RemoveUnusedTensors(),
        SortGraph(),
    ]:
        model = model.transform(trn)

    _assert_no_adds(model, "pre-input clean")
    return _save_and_check(model, cfg, "00_pre_input_clean")


@step(name="post_qonnx_to_finn")
def step_vgg16_post_qonnx_to_finn_check(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Validate immediately after step_qonnx_to_finn.

    Expected:
      - QONNX Quant/Trunc/BinaryQuant nodes should have been converted.
      - Activations should be represented in FINN-friendly form, often
        MultiThreshold.
    """
    model = _tidy(model)

    _assert_no_ops(
        model,
        ["Quant", "Trunc", "BinaryQuant"],
        "step_qonnx_to_finn",
    )
    _assert_no_adds(model, "step_qonnx_to_finn")

    return _save_and_check(model, cfg, "01_post_qonnx_to_finn")


@step(name="pre-streamline")
def step_vgg16_pre_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    VGG16-specific cleanup before streamlining.

    Do not use Change3DTo4DTensors here: VGG16 image input is already 4D NCHW.
    """
    for trn in [
        InferShapes(),
        FoldConstants(),
        DoubleToSingleFloat(),
        GemmToMatMul(),
        FoldTransposeIntoQuantInit(),
        InferDataTypes(),
        InferDataLayouts(),
        AbsorbConsecutiveTransposes(),
        RemoveStaticGraphInputs(),
        RemoveUnusedTensors(),
        SortGraph(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
    ]:
        model = model.transform(trn)

    _assert_no_adds(model, "pre-streamline")
    return _save_and_check(model, cfg, "02_pre_streamline")


@step(name="post-streamline")
def step_vgg16_post_streamline_check(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Cleanup and validate after streamlining.

    This does not ban all Transpose/Add/Mul nodes because some scalar/layout
    nodes can still be legal at this point. It specifically rejects residual
    Add branches, which VGG16 should not contain.
    """
    for trn in [
        InferShapes(),
        InferDataLayouts(),
        InferDataTypes(),
        
        CollapseRepeatedAdd(),
        CollapseRepeatedMul(),
        MoveAddPastConv(),
        MoveScalarMulPastConv(),
        MoveScalarAddPastMatMul(),
        MoveScalarMulPastMatMul(),
        MoveScalarLinearPastInvariants(),
        
        
        AbsorbAddIntoMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        AbsorbTransposeIntoMultiThreshold(),
        
        AbsorbConsecutiveTransposes(),
        AbsorbTransposeIntoMultiThreshold(),
        
        RemoveIdentityOps(),
        RemoveUnusedTensors(),
        SortGraph(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
    ]:
        model = model.transform(trn)

    _assert_no_adds(model, "post-streamline")
    return _save_and_check(model, cfg, "03_post_streamline")


@step(name="post_convert_to_hw_check")
def step_vgg16_post_convert_to_hw_check(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Validate after step_convert_to_hw.

    If Conv or Gemm remains here, conversion to FINN HW layers is incomplete.
    MatMul/Im2Col may or may not remain depending on exactly which FINN/Brainsmith
    built-in steps have already run, so this check keeps the failure condition
    strict but not over-specific.
    """
    model = _tidy(model)

    _assert_no_ops(model, ["Conv", "Gemm"], "step_convert_to_hw")
    _assert_no_adds(model, "post_convert_to_hw_check")

    return _save_and_check(model, cfg, "04_post_convert_to_hw_check")


@step(name="convert_optional_hw_layers")
def step_vgg16_convert_optional_final_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Optional conversion for final channelwise/label-select style tail layers.

    This is harmless if those patterns are absent.
    """
    for trn in [
        RemoveCNVtoFCFlatten(),
        
        AvgPoolAndTruncToQuantAvgPool(),
        to_hw.InferPool(),
        #to_hw.InferStreamingMaxPool(),
        to_hw.InferGlobalAccPoolLayer(),
        
        to_hw.InferQuantizedMatrixVectorActivation(),
        to_hw.InferThresholdingLayer(),
        to_hw.InferConvInpGen(),
        
        RemoveCNVtoFCFlatten(),
        RoundAndClipThresholds(),
        
        to_hw.InferChannelwiseLinearLayer(),
        to_hw.InferLabelSelectLayer(),
        
        AbsorbConsecutiveTransposes(),
        InferShapes(),
        InferDataTypes(),
        InferDataLayouts(),
        RemoveUnusedTensors(),
        SortGraph(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
    ]:
        model = model.transform(trn)

    _assert_no_adds(model, "convert_optional_hw_layers")

    def is_fpgadataflow_node(node):
        backend = get_by_name(node.attribute, "backend")
        return backend is not None and backend.s.decode("UTF-8") == "fpgadataflow"

    nodes = list(model.graph.node)
    df_idxs = [i for i, n in enumerate(nodes) if is_fpgadataflow_node(n)]

    print("First/last fpgadataflow node:", min(df_idxs), max(df_idxs))

    print("\nNon-fpgadataflow nodes inside fpgadataflow span:")
    for i in range(min(df_idxs), max(df_idxs) + 1):
        n = nodes[i]
        if not is_fpgadataflow_node(n):
            print(i, n.name, n.op_type, "domain=", n.domain)

    return _save_and_check(model, cfg, "05_convert_optional_hw_layers")


@step(name="post_dataflow_partition_check")
def step_vgg16_post_dataflow_partition_check(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Check after create_dataflow_partition.

    FINN often saves the parent graph as:
        <output_dir>/intermediate_models/dataflow_parent.onnx

    and continues with the child dataflow graph. This step checks the parent
    when present and always saves/checks the current child graph.
    """
    intermediate_dir = os.path.join(cfg.output_dir, "intermediate_models")
    parent_path = os.path.join(intermediate_dir, "dataflow_parent.onnx")

    if os.path.isfile(parent_path):
        parent_model = ModelWrapper(parent_path)
        sdp_count = _count_ops(parent_model, "StreamingDataflowPartition")

        if sdp_count != 1:
            raise RuntimeError(
                f"Expected exactly one StreamingDataflowPartition in parent graph, "
                f"found {sdp_count}."
            )

        _assert_no_adds(parent_model, "dataflow parent partition")
        _save_and_check(parent_model, cfg, "06_post_dataflow_partition_parent")
    else:
        sdp_count = _count_ops(model, "StreamingDataflowPartition")

        if sdp_count not in [0, 1]:
            raise RuntimeError(
                f"Expected zero or one StreamingDataflowPartition in current graph, "
                f"found {sdp_count}."
            )

    _assert_no_adds(model, "post_dataflow_partition_check_child")
    return _save_and_check(model, cfg, "07_post_dataflow_partition_child")


# -----------------------------------------------------------------------------
# Optional VGG16 manual streamlining / HW inference steps
# -----------------------------------------------------------------------------

@step(name="vgg16_streamline")
def vgg16_manual_streamline_step(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Optional manual streamline step.

    Use only if the blueprint does not call the built-in step_streamline.
    """
    for trn in [
        MoveScalarLinearPastInvariants(),
        Streamline(),
        AbsorbConsecutiveTransposes(),
        AbsorbTransposeIntoMultiThreshold(),
        RemoveUnusedTensors(),
        SortGraph(),
    ]:
        model = model.transform(trn)

    _assert_no_adds(model, "vgg16_manual_streamline")
    return _save_and_check(model, cfg, "manual_streamline")


@step(name="vgg16_clean_transposes_before_hw")
def vgg16_clean_transposes_before_hw_step(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Optional transpose cleanup before hardware-layer inference.

    Use this if transpose/layout nodes are blocking dataflow partitioning.
    """
    _print_transposes(model, "initial")

    for trn in [
        AvgPoolAndTruncToQuantAvgPool(), #
        to_hw.InferPool(), #
        #to_hw.InferStreamingMaxPool(), #
        to_hw.InferGlobalAccPoolLayer(), # <---
        
        FoldTransposeIntoQuantInit(),
        InferDataLayouts(),
        AbsorbConsecutiveTransposes(),
        MoveScalarLinearPastInvariants(),
        AbsorbTransposeIntoMultiThreshold(),
        AbsorbConsecutiveTransposes(),
    ]:
        model = model.transform(trn)
        model = _tidy(model)

    if _count_ops(model, "MaxPool") > 0:
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(AbsorbConsecutiveTransposes())
        model = _tidy(model)

    # Expose Conv as Im2Col/MatMul if this has not already been done by a
    # built-in Brainsmith/FINN step.
    if _count_ops(model, "Conv") > 0:
        model = model.transform(LowerConvsToMatMul())
        model = _tidy(model)

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = _tidy(model)

    _print_transposes(model, "after cleanup")
    _assert_no_adds(model, "vgg16_clean_transposes_before_hw")

    return _save_and_check(model, cfg, "manual_clean_transposes_before_hw")


@step(name="vgg16_infer_hw_layers")
def vgg16_manual_infer_hw_layers_step(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Optional manual HW-layer inference step.

    Use only if the blueprint does not call the built-in step_convert_to_hw.
    """
    for trn in [
        LowerConvsToMatMul(),
        RemoveCNVtoFCFlatten(),
        RoundAndClipThresholds(),

        InferShapes(),
        InferDataTypes(),
        to_hw.InferQuantizedMatrixVectorActivation(),
        to_hw.InferThresholdingLayer(),
        to_hw.InferConvInpGen(),

        to_hw.InferPool(),
        #to_hw.InferStreamingMaxPool(),
        to_hw.InferGlobalAccPoolLayer(),
        
        #to_hw.InferDuplicateStreamsLayer(),
        to_hw.InferVectorVectorActivation(),
        to_hw.InferChannelwiseLinearLayer(),
        to_hw.InferLabelSelectLayer(),
        InferShapes(),
        InferDataTypes(),
        InferDataLayouts(),
        AbsorbConsecutiveTransposes(),

        RemoveUnusedTensors(),
        SortGraph(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
    ]:
        model = model.transform(trn)

    _assert_no_adds(model, "vgg16_manual_infer_hw_layers")
    return _save_and_check(model, cfg, "manual_infer_hw_layers")


@step(name="vgg16_specialize_remaining_hw_layers")
def vgg16_specialize_remaining_hw_layers_step(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
):
    """
    Run after build_hw_graph and before minimize_bit_width / estimate reports.
    """   

    # Resolve FPGA part. Pynq-Z1 uses xc7z020clg400-1.
    if hasattr(cfg, "_resolve_fpga_part"):
        fpgapart = cfg._resolve_fpga_part()
    else:
        fpgapart = getattr(cfg, "fpga_part", None)

    if fpgapart is None:
        # Safe fallback for Pynq-Z1 / Zynq-7020
        fpgapart = "xc7z020clg400-1"

    print("\n[vgg16_specialize_remaining_hw_layers] FPGA part:", fpgapart)

    def get_backend(node):
        backend_attr = get_by_name(node.attribute, "backend")
        if backend_attr is None:
            return None
        return backend_attr.s.decode("UTF-8")

    def is_generic_fpgadataflow_node(node):
        # Generic FINN fpgadataflow nodes often have this domain and no _hls/_rtl suffix.
        if node.domain != "finn.custom_op.fpgadataflow":
            return False

        if node.op_type.endswith("_hls") or node.op_type.endswith("_rtl"):
            return False

        return True

    print("\n[vgg16_specialize_remaining_hw_layers] Before specialization:")
    generic_before = []

    for node in model.graph.node:
        backend = get_backend(node)

        if is_generic_fpgadataflow_node(node):
            generic_before.append((node.name, node.op_type, node.domain, backend))
            print(
                f"  generic node still present: "
                f"{node.name} :: {node.op_type} :: domain={node.domain} :: backend={backend}"
            )

    if len(generic_before) == 0:
        print("  No generic fpgadataflow nodes found before specialization.")

    # Re-run FINN specialization.
    model = model.transform(SpecializeLayers(fpgapart))
    model = _tidy(model)

    print("\n[vgg16_specialize_remaining_hw_layers] After specialization:")
    generic_after = []

    for node in model.graph.node:
        backend = get_backend(node)

        if is_generic_fpgadataflow_node(node):
            generic_after.append((node.name, node.op_type, node.domain, backend))
            print(
                f"  still generic: "
                f"{node.name} :: {node.op_type} :: domain={node.domain} :: backend={backend}"
            )

    if len(generic_after) == 0:
        print("  No generic fpgadataflow nodes remain after specialization.")

    # Attach cycle estimates where possible.
    # This is useful because generate_estimate_reports/dataflow_performance
    # depends on expected latency/cycle information.
    try:
        model = model.transform(AnnotateCycles())
    except Exception as e:
        print(
            "[vgg16_specialize_remaining_hw_layers] "
            f"AnnotateCycles failed or is unsupported for some node: {repr(e)}"
        )

    # Save for inspection.
    out_dir = _custom_out_dir(cfg)
    model_path = os.path.join(out_dir, "13b_vgg16_specialized_remaining_hw_layers.onnx")
    model.save(model_path)
    print(f"[vgg16_specialize_remaining_hw_layers] saved: {model_path}")

    # Hard check only for the ops that are causing your current failure.
    # If this raises, then your installed FINN version likely does not provide
    # an HLS/RTL specialization for FMPadding.
    remaining_problem_nodes = [
        item for item in generic_after
        if item[1] in ["FMPadding", "ConvolutionInputGenerator", "Pool"]
    ]

    if len(remaining_problem_nodes) > 0:
        raise RuntimeError(
            "Generic fpgadataflow nodes still remain after SpecializeLayers. "
            "This likely means your installed FINN version does not provide "
            "an HLS/RTL specialization for one or more of these ops, or the "
            "node is intended to remain generic. Remaining nodes: "
            + str(remaining_problem_nodes)
        )

    return model
