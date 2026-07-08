# Tutorials

Refer to the examples and the requirements to execute the build flow.

# Build Pre-requisites

1. Blueprint file (.yaml): defines the hardware design space definition with inheritance.
2. Quantized ONNX graph (.onnx): quantization parameters (bit widths, scales, thresholds) are baked into the model structure, making it hardware-aware, and used for Intermediate Representation (IR).​
3. Python Model build/import script (.py): loads the trained model (typically from PyTorch/Brevitas) and exports it into the ONNX format with quantization operators embedded in the graph.

# 1. Blueprint file

 This file acts as the 'blueprint' or the template of the neural network accelerator design, which includes - the kernels, transformation steps, and build parameters.
 
 ### Supported boards:

| Board Set Name | Type | URL |
|---|---|---|
| avnet-boards | git | https://github.com/Avnet/bdf.git |
| xilinx-boards | git | https://github.com/Xilinx/XilinxBoardStore.git |
| rfsoc4x2-boards | git | https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git |
| kv260-som-boards | git | https://github.com/Xilinx/XilinxBoardStore.git |
| aupzu3-boards | git | https://github.com/RealDigitalOrg/aup-zu3-bsp.git |
| pynq-z1 | zip | https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip |
| pynq-z2 | zip | https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip |

 Refer the following [link](https://github.com/microsoft/brainsmith/blob/6b1e9ef1bee0ce63561d32ccbf16b33ae2cdff80/docs/developer-guide/experimental/3-reference/blueprints.md) to understand the blueprint file generation.

# 2. Pytorch script

This script should contain the neural network architecuture either imported from platforms like huggingface or created using modelwrappers from QONNX.
The following are the scripts utilised to compile and generate the streaming dataflow accelerator:
- model import/build script
- custom steps script

# 3. ONNX file

The quantized ONNX file is generated as a result of the compilation of the model import/build script, which is constructed using QONNX and Brevitas.



