---
layout: default
title: Brainsmith Compiler Log
---

# Brainsmith

Brainsmith is a compiler created by AMD and Microsoft. It transforms ONNX models of neural networks, using automated design space exploration, into optimized streaming dataflow accelerators for deploying onto FPGAs for neral network inference. In this repository, we aimed to generate RTL design and the final stitched version of the LeNet-5 Model.

Key stages in generating the final output include:
- Model transformation: Converting ONNX operations to hardware kernels
- Design space exploration: Determining parallelization factors (PE/SIMD)
- Code generation: HLS (C++) and/or RTL (hardware code generation)
- IP packaging: Creating Vivado IP Cores
- Simulation: Verifying correctness with RTL simulation in Vivado (of RTL file/bitfile)

## Quick Links
- [Installation Guide](/docs/setup)
- [Getting Started](/docs/usage)
