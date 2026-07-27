# VGG16 Model Accelerator Design

## Model Features

| Parameter          | Default_values | Signifies                                                          |
| ------------------ | ------- | ----------------------------------------------------------------------------- |
| `in_channels`      | `1`     | Number of input channels in the image.                                        |
| `num_classes`      | `1000`    | Number of output classes the model predicts.                                  |
| `act_bit_width`    | `8`     | Number of bits used to represent activation values during inference/training. |

- This model is trained on the mnist datatset
- The model accepts input images/vectors of dimensionality (1,3,224,224)
- We may also mention the number of classes and the activation bit width along with the compile command with the syntax:

      python model.py --blueprint <blueprint>.yaml --output <output_file_path> --num-classes <value> --bit-width <value>
- The default location where the onnx files generated as a result of the execution of the VGG16 and the other examples are with the file path: ._/onnx/<model>_raw.onnx_ and _./onnx/<model>_clean.onnx_ which could also be altered while writing the compile command with "--raw-onnx" and "--clean-onnx" as the optional command line arguements.
- All the customisations required could be performed upon the model.py script which is used to build our model.
- We have used the command:

      python model.py --blueprint vggnet.yaml --output ../results/run1
  which firstly generates the ONNX files - raw and the cleaned files, automating to proceed with creation of streaming dataflow accelerator.

## Custom steps defined:

| Custom Step                              | Pipeline Stage                      | Primary Purpose                                                                    | Main Transformations Performed                                                                                                          |
| ---------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | 
| **pre-input clean**                      | Before `qonnx_to_finn`              | Cleans the exported QONNX graph while preserving quantization information          | Removes redundant tensors, folds constants, converts GEMM→MatMul, normalizes datatypes/layouts, removes identity operators, sorts graph | 
| **post_qonnx_to_finn**                   | Immediately after `qonnx_to_finn`   | Verifies successful conversion from QONNX to FINN representation                   | Checks that `Quant`, `BinaryQuant`, and `Trunc` operators have disappeared and performs graph cleanup                                   | 
| **pre-streamline**                       | Before streamlining                 | Performs graph normalization before optimization                                   | Constant folding, datatype/layout inference, transpose absorption, graph cleanup                                                        | 
| **post-streamline**                      | After streamlining                  | Validates the optimized graph                                                      | Collapses repeated Add/Mul operations, moves scalar operations, absorbs threshold operations, removes identities                        | 
| **post_convert_to_hw_check**             | After hardware conversion           | Confirms that software operators were converted into hardware-compatible operators | Verifies that `Conv` and `Gemm` no longer exist                                                                                         | 
| **convert_optional_hw_layers**           | Before partitioning                 | Converts remaining operators that are not always handled automatically             | Converts pooling, global accumulation pool, thresholding, label select, channelwise linear layers, Conv input generators                | 
| **post_dataflow_partition_check**        | After partitioning                  | Validates the generated hardware partition                                         | Checks that exactly one `StreamingDataflowPartition` exists and validates parent/child graphs                                           | 
| **vgg16_streamline** *(optional)*        | Replacement for built-in streamline | Manual streamline implementation                                                   | Executes FINN `Streamline`, transpose absorption, graph sorting                                                                         | 
| **vgg16_clean_transposes_before_hw**     | Before HW inference                 | Eliminates transpose/layout issues that interfere with hardware inference          | Removes transpose chains, converts pooling, lowers convolutions to MatMul, removes Conv→FC flatten operations                           |
| **vgg16_infer_hw_layers**                | Before HW inference                  | Converts remaining neural-network layers into FINN hardware operators              | Infers MVAU, Thresholding, ConvolutionInputGenerator, Pool, ChannelwiseLinear, LabelSelect, etc.                                        | 
| **vgg16_specialize_remaining_hw_layers** | Hardware inference           | Converts generic hardware operators into FPGA-specific HLS/RTL implementations     | Runs `SpecializeLayers`, annotates cycle estimates, checks that no unsupported generic operators remain                                 | 


**For any such similar model these custom steps are required because:**

1. To prepare and normalize the exported QONNX graph.
2. Validate that each transformation stage completes correctly.
3. Remove layout and transpose patterns that hinder hardware conversion.
4. Infer all FPGA-compatible hardware operators.
5. Verify successful dataflow partitioning. (the most common issue with CNNs and large transformer Models)
6. Specialize generic hardware operators into FPGA-specific HLS/RTL implementations specific to the FPGA resource
7. Save intermediate ONNX models and diagnostic reports to simplify debugging when a later synthesis stage (such as hw_codegen or hw_ipgen) fails.
