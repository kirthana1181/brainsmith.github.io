# Introduction

Brainsmith automates the construction of dataflow accelerators, handling kernel selection, FIFO sizing, parallelization, and IP generation, making FPGA acceleration accessible without deep hardware expertise, and is composed 3 foundational open-source tools in the order: Brevitas, QONNX and the FINN Compiler.

# Installation requirements for Brainsmith Compiler

1. Ubuntu 22.04+ (primary development/testing platform)
2. Vivado Design Suite 2024.2
3. Cmake for V80 shell integration (optional, depends on the applications target HW resource)
4. Python 3.11.*
   
Ensure 'smith' command line has been installed, which is used to run the compiler and create our streamlined dataflow-accelerators (DFA).
