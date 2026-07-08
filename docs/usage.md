# Basic Commands to run the Brainsmith Compiler

`smith` - Streamlined CLI for creating dataflow accelerators

`brainsmith` - Full toolkit with administrative commands

| Command | Description |
|---------|-------------|
| `brainsmith project init` | Initialize a new Brainsmith project with default configuration files |
| `brainsmith registry` | List all registered components and available dataflow accelerators |
| `brainsmith setup cppsim` | Setup and configure C++ simulation environment for testing |

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
- **`estimate_layer_cycles.json`** - Estimated clock cycles required per layer
- **`estimate_layer_resources.json`** - FPGA resource utilization estimates (DSP, BRAM, LUT)
- **`estimate_network_performance.json`** - Overall network performance metrics
- **`op_and_param_counts.json`** - Operation and parameter statistics
- **`rtlsim_performance.json`** - RTL simulation performance results
