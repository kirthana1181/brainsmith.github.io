---
layout: default
title: Brainsmith
---

[Setup]('docs/setup.md') | [Tutorial]('docs/tutorial.md') | [CLI]('docs/usage.md')

---

# Home

---

Brainsmith is a compiler created by AMD and Microsoft. It transforms ONNX models of neural networks, using automated design space exploration, into optimized streaming dataflow accelerators for deploying onto FPGAs for neral network inference. In this repository, we aimed to generate RTL design and the final stitched version of the LeNet-5 Model.

Key stages in generating the final output include:
- Model transformation: Converting ONNX operations to hardware kernels
- Design space exploration: Determining parallelization factors (PE/SIMD)
- Code generation: HLS (C++) and/or RTL (hardware code generation)
- IP packaging: Creating Vivado IP Cores
- Simulation: Verifying correctness with RTL simulation in Vivado (of RTL file/bitfile)

Key features:

**Automated Design** - Handles kernel selection and IP generation  
**Optimized Performance** - FIFO sizing and parallelization  
**Easy Integration** - Seamless Vivado compatibility  
 

# Installation requirements for Brainsmith Compiler

1. Ubuntu 22.04+ (primary development/testing platform)
2. Vivado Design Suite 2024.2
3. Cmake for V80 shell integration (optional, depends on the applications target HW resource)
4. Python 3.11.*
   
Ensure 'smith' command line has been installed, which is used to run the compiler and create our streamlined dataflow-accelerators (DFA).

# Understanding Brainsmith

Brainsmith is nothing but a tool stitched out of these open-source tools: Brevitas, QONNX and FINN Compiler, to build stitched IP of large neural networks for deployment of high-performance neural network accelerator design by specializing hardware through customizable RTL generation and dataflow modeling.

1. Train a custom quantized neural network (QNN) in Brevitas. Follow how to do quantization aware training (QAT) using [Brevitas](https://xilinx.github.io/brevitas/v0.12.1/tutorials/tvmcon2021.html).
2. Export your model to [QONNX](https://qonnx.readthedocs.io/en/latest/).
3. Use FINN build_dataflow functionality on the exported model following this [link](https://github.com/Xilinx/finn/blob/main/src/finn/builder/build_dataflow_steps.py) to write down the required steps in the blueprint(.yaml) or for a complex builder settings follow this [tutorial](https://github.com/Xilinx/finn/blob/main/notebooks/advanced/4_advanced_builder_settings.ipynb)
4. Tweak your QNN topology, quantization setup and build_dataflow parameters to obtain the desired outcome.
5. Define and state the build dataflow steps as per the neural network required to be deployed, using [FINN and QONNX](https://finn.readthedocs.io/en/latest/source_code/finn.transformation.html) transformations, and execute the compiler command to build the streaming dataflow accelerator.

### [Getting Started](/docs/usage)

## Quick Links
- [Start Building]('/docs/setup') 
- [Tutorial]('/docs/tutorial')
- [CLI]('/docs/usage')
