---
layout: default
title: Brainsmith
---

[Home](docs/index.md) | [Setup](docs/setup.md) | [Tutorial](docs/tutorial.md) | [CLI](docs/usage.md)

---

# CLI
...
# Basic Commands to run the Brainsmith Compiler

`smith` - Streamlined CLI for creating dataflow accelerators

`brainsmith` - Full toolkit with administrative commands

| Command | Description |
|---------|-------------|
| `brainsmith project init` | Initialize a new Brainsmith project with default configuration files |
| `brainsmith registry` | List all registered components and available dataflow accelerators |
| `brainsmith setup cppsim` | Setup and configure C++ simulation environment for testing |

Refer to this [github file](https://github.com/microsoft/brainsmith/blob/6b1e9ef1bee0ce63561d32ccbf16b33ae2cdff80/docs/api/cli.md) which explains further commands in Brainsmith CLI.

------------------------------------------------------------------------

1. Running the compiler: generates the streaming dataflow accelerator.

       smith dfc model.onnx --blueprint.yaml --options <input>
   
| options | input |
|---------|-------|
| `start-step` | `blueprint_step_name` |
| `stop-step` | `blueprint_step_name` |
| `output-dir` | `output_file_path` |
   
   
2. Command to create project in brainsmith and activate environment

       #Activate brainsmith venv if not in an active project
       source /path/to/brainsmith/.venv/bin/activate
   
   OR
   
       #Activate brainsmith venv if not in an active project
       cd /path/to/brainsmith/ && source .venv/bin/activate

        #Create and initialize new project directory
        brainsmith project init <project_name>
        cd <project_name>

3. Resulting reports:

   The reports generated depend on the type of final output ('estimates', 'rtl' or 'bitfile')

- **`estimate_layer_config_alternatives.json`** - Alternative hardware configurations for each layer
- **`estimate_layer_cycles.json`** - Represents the number of cycles taken by each of the output ONNX nodes (i.e. each layer)
- **`estimate_layer_resources.json`** - Represents the amount of HW resources required specific to the output ONNX nodes, within the target FPGA resource​ estimates
- **`estimate_network_performance.json`** - Summarizes the estimated end-to-end performance, highlighting performance metrics
- **`op_and_param_counts.json`** - MAC operation and number of weight parameters statistics specific to each ONNX node
- **`rtlsim_performance.json`** - RTL simulation performance results, generated as a result of the step ‘measure_rtl_sim_performance’
