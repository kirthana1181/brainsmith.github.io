---
title: Tutorials
layout: default
---

Refer to the [examples](https://github.com/kirthana1181/brainsmith.github.io/tree/main/examples) and the requirements to execute the build flow.
The basic build flow is as shown:

    PyTorch → ONNX → Hardware Kernels → HLS/RTL → IP Cores → Bitfile

# Build Pre-requisites

1. Blueprint file (.yaml): defines the hardware design space definition with inheritance.
2. Quantized ONNX graph (.onnx): quantization parameters (bit widths, scales, thresholds) are baked into the model structure, making it hardware-aware, and used for Intermediate Representation (IR).​
3. Python Model build/import script (.py): loads the trained model (typically from PyTorch/Brevitas) and exports it into the ONNX format with quantization operators embedded in the graph.
4. Custom Steps scipt (.py) (optional) : defines and exports custom build steps, which are defined specific to the application(Model), using QONNX and FINN transformations over the ONNX file of the application/Model.

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

The build script we saved in the case of our examples is named as _model.py_ , and the custom steps file is named as _custom_steps.py_. The python script which defines the folding configuration w.r.t the Output onnx file is described in the specific to the output onnx graph _gen_folding.py_.

# 3. ONNX file

The quantized ONNX file is generated as a result of the compilation of the model import/build script, which is constructed using QONNX and Brevitas libraries. Both the origin and cleaned up ONNX graphs are generated using this build script, with the help of transformations from the custom_steps script.

## Steps to compile and create the streaming dataflow accelerator in Brainsmith:

1. Enable the virtual environment using the following command or similar:

       source .venv/bin/activate

   OR

       python -m venv venv
3. Define the project directory using the command:

       brainsmith project init <project name>
   
   In our LeNet-5 example, we have named the project as the Model we are trying to implement i.e. '**lenet**'.

4. Enable the environment using the command: "_**direnv allow**_".
5. Define the model bluprint, model build and custom steps scripts.
     - Use [this link(1)](https://github.com/microsoft/brainsmith/blob/main/examples/blueprints/base.yaml) to view the basic blueprint file template and steps.
     - Refer to [this link(2)](https://github.com/Xilinx/finn/blob/main/src/finn/builder/build_dataflow_steps.py) to view the description of basic dataflow steps, to be stated under the blueprint's "**design_space**" section.
     - Write the build and custom steps script using pytorch and the Brevitas and FINN libraries, which are used to define the custom steps in the latter python script.
     - Both the origin and cleanup ONNX files should be generated as result of the compilation of the model build script. Optionally we may include the build flow, using the brainsmith registry and dse libraries.

           from brainsmith.dse.api import explore_design_space
           from brainsmith.registry import has_step
     -  This replaces the need to explicitly run the brainsmith compile command using 'smith':

            smith dfc <cleaned-up_onnx_file> <blueprint_file> <optional-arguments>
        where the optional arguements could possibly be:
        
          | Option	| Type	| Default	| Description |
          |-------|-------|-------|-------|
          |-o,--output-dir |	Path	| build/{timestamp} |	Output directory for generated files |
          |--start-step	| Text | -| Override blueprint | start_step (start execution from this step, inclusive) |
          --stop-step	| Text |	-|	Override blueprint | stop_step (stop execution at this step, inclusive |
     
            
6. Use the following command to view the project directory information:

       brainsmith project info


## Generated Outputs

The following outputs will be generated regardless of which particular outputs are selected:

- build_dataflow.log is the build logfile that will contain any warnings/errors
- time_per_step.json will report the time (in seconds) each build step took
- final_hw_config.json will contain the final (after parallelization, FIFO sizing etc) hardware configuration for the build. It is written by the FIFO sizing step, so it is not produced for estimate-only builds (where FIFO sizing is skipped)
- template_specialize_layers_config.json is an example json ile that can be used to set the specialize layers config.
- intermediate_models/ will contain the ONNX file(s) produced after each build step.

---

[Setup]({{ "/docs/setup/" | relative_url }})
|
[CLI]({{ "/docs/usage/" | relative_url }})

---


