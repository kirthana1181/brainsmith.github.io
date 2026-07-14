# BERT Example for Brainsmith
_Referred from [github documentation site (3)](https://github.com/microsoft/brainsmith?tab=contributing-ov-file)_

This example demonstrates accelerating BERT transformer models on FPGA, showcasing Brainsmith's ability to handle complex neural networks through automated design space exploration.

## Overview

The BERT example shows how to:
- Generate quantized BERT models using Brevitas
- Extract the acceleratable transformer core 
- Apply custom transformations for FPGA optimization
- Generate RTL and bitfiles through Brainsmith's DSE pipeline
- Configure hardware parallelism with folding parameters

## Quick Start

Run a minimal 1-layer BERT test:
```bash
python bert_demo.py --blueprint bert_quicktest.yaml -o bert_demo_output  -l 1 
```

## Prerequisites

- Brainsmith development environment (via `smithy` container) (In our case, here we did not have smithy installed, so python works!)
- Xilinx Vivado 2024.2 (for bitfile generation)

## Usage

### Basic Command
```bash
./smithy python ./examples/bert/bert_demo.py -o my_bert_build
```

### Custom Model Configuration
```bash
# Small BERT with 4-bit quantization
python bert_demo.py \
    --blueprint bert_quicktest.yaml \
    -o bert_demo_output \
    -l 1 \
    -z 64 \
    -n 4 \
    -b 4 \

# Larger model with custom blueprint
python bert_demo.py \
    --blueprint bert_quicktest.yaml \
    -o bert_demo_output \
    -l 1 \
    -z 64 \
    -n 4 \
    -i 768 \
    -b 4 \
    -q 32 \
    --blueprint bert_quicktest.yaml
```

### Command-Line Options
- `-o, --output`: Output directory name (required)
- `-z, --hidden_size`: BERT hidden dimension (default: 384)
- `-n, --num_attention_heads`: Number of attention heads (default: 12)
- `-l, --num_hidden_layers`: Number of transformer layers (default: 1)
- `-i, --intermediate_size`: FFN intermediate size (default: 1536)
- `-b, --bitwidth`: Quantization bits {4,8} (default: 8)
- `-q, --seqlen`: Sequence length (default: 128)
- `--blueprint`: Blueprint YAML file (default: bert_demo.yaml)

## Blueprint Configuration

The example includes two blueprint configurations:

### bert_demo.yaml
- Full synthesis and implementation flow
- Targets 3000 FPS performance
- Automatic folding configuration
- Complete bitfile generation

### bert_quicktest.yaml
- Optimized for quick iteration
- Uses pre-generated folding config
- Lower performance target (1 FPS)
- Skips time-intensive optimizations

## Custom Steps (Runtime Component Registration)

This example demonstrates how to create custom steps using the `@step` decorator
without needing a full package structure. The steps in `custom_steps.py`
are registered when imported by `bert_demo.py` and referenced by name in the
blueprint YAML.

**Local steps defined in custom_steps.py:**
- **remove_head**: Extracts transformer encoder by removing embedding layers
- **remove_tail**: Removes classification head to focus on encoder
- **generate_reference_io**: Creates test vectors for RTL verification

**Core brainsmith steps used from brainsmith.steps.bert_custom_steps:**
- **bert_cleanup**: BERT-specific model cleanup and normalization
- **bert_streamlining**: Streamline BERT model structure
- **shell_metadata_handover**: Extract metadata for shell integration

All steps are referenced in `bert_demo.yaml` and executed as part of the
dataflow compilation pipeline. This pattern is ideal for example-specific
transformations that don't need to be shared across projects.

## Folding Configuration

Control hardware parallelism using `gen_folding_config.py`:

```bash
# Generate custom folding config
python gen_folding_config.py \
    --pe 8 \
    --simd 8 \
    --output configs/my_folding.json
```

Folding parameters determine the PE×SIMD parallelism for each layer, directly affecting resource usage and throughput.
