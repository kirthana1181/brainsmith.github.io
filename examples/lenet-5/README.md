# LeNet-5 Accelerator Design

## Features of the Model:

| Parameter          | Default_values | Signifies                                                          |
| ------------------ | ------- | ----------------------------------------------------------------------------- |
| `in_channels`      | `1`     | Number of input channels in the image.                                        |
| `num_classes`      | `10`    | Number of output classes the model predicts.                                  |
| `weight_bit_width` | `8`     | Number of bits used to represent the model's weights (quantization).          |
| `act_bit_width`    | `8`     | Number of bits used to represent activation values during inference/training. |


- The Model only intakes images/vetors with the dimensionality - (1,1,28,28) _(Note: this vector of the **NCHW** format)_
- These values can be modified within the model import/build script.
- The model is built using Modelwrapper which utilises pooling in convolutional layers in place of the regular MaxPooling or AvgPooling layers in the intermediate of the CNN.

## LeNet-5 Custom-defined build steps

| Custom Step                            | Purpose                                                                                                                                                                     |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lenet_pre_dataflow_cleanup`           | Performs safe cleanup before custom hardware conversion.                                                                                                                    |
| `lenet_streamline`                     | Performs safe cleanup before custom hardware conversion and streamlines the FINN/QONNX graph before hardware-layer inference.                                               |
| `lenet_clean_transposes_before_hw`     | Specifically looks up floating transpose nodes in the graph, quantizes them, and converts them to `MultiThreshold` nodes.                                                   |
| `lenet_infer_hw_layers`                | Significant to CNNs; converts the convolution, threshold, and matmul nodes into `fpgadataflow` nodes.                                                                       |
| `lenet_specialize_remaining_hw_layers` | Inserted after `build_hw_graph`; converts any remaining generic FINN nodes such as `FMPadding` and `ConvolutionInputGenerator`, as in our graph, into `fpgadataflow` nodes. |
| `lenet_pre_partition_cleanup`          | Removes ordinary `Transpose`/`Reshape` nodes trapped between FINN hardware nodes.                                                                                           |

## Results

As a result of the implementation, we conducted experiments over the LeNet-5 Model to confirm the trends in PE and SIMD scaling factors vs. latency cyles and Hardware resources.

![Latency Cycles vs. Resources(PE)](/assets/images/LatencyCycles_vs_Resources(PE).png)
Latency Cycles vs. Resources(PE)

![Latency Cycles vs. Resources(SIMD)](/assets/images/LatencyCycles_vs_Resources(SIMD).png)
Latency Cycles vs. Resources(SIMD)

![Throughput vs. Resources(PE)](/assets/images/Throughput_vs_Resources(PE).png)
Throughput vs. Resources(PE)

![Throughput vs. Resources(SIMD)](/assets/images/Throughput_vs_Resources(SIMD).png)
Throughput vs. Resources(SIMD)

RTL Simulation Performance Results at 200MHz:

```json
  {
  "N_IN_TXNS": 784,
  "N_OUT_TXNS": 10,
  "cycles": 128319,
  "N": 1,
  "latency_cycles": 128318,
  "interval_cycles": 128308,
  "TIMEOUT": 0,
  "UNFINISHED_INS": 0,
  "UNFINISHED_OUTS": 0,
  "RUNTIME_S": 6,
  "runtime[ms]": 0.641595,
  "throughput[images/s]": 1558.615637590692,
  "fclk[mhz]": 200.0,
  "stable_throughput[images/s]": 1558.615637590692
}
```

_Refer to this [github repository link](https://github.com/kirthana1181/LeNet-5-using-Brainsmith.git) to understand the LeNet-5 implementation._
