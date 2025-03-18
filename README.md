# Benchmarking RTX5090D based on Timm Models

This README provides a comparison of the training and inference speeds of various deep learning models using different precisions (FP32 and FP16) on RTX5090D configurations. We compare the results obtained from our benchmarking script with the results published in [RTX5090 Benchmark](https://nikolasent.github.io/hardware/deeplearning/benchmark/2025/02/17/RTX5090-Benchmark.html) .

## Benchmark Results

### FP32 Training Speeds
| Model Name | RTX 3090 | RTX 4090 | RTX 5090 | RTX5090D | RTX5090D vs RTX5090 |
| --- | --- | --- | --- | --- | --- |
| VGG16 | 260.8 | 456.5 (+75.1%) | 594.7 (+128.1%) | 581.1 | -2.3% |
| ResNet50 | 523.0 | 757.5 (+44.8%) | 1128.6 (+115.8%) | 1113.0 | -1.4% |
| TF EfficientNetV2 B0 | 1145.8 | 1643.8 (+43.5%) | 2448.9 (+113.7%) | 2215.6 | -9.5% |
| Swin Base Patch4 Window7 224 | 158.0 | 293.2 (+85.6%) | 450.2 (+185.0%) | 261.6 | -41.9% |
| EfficientViT M4 | 2730.0 | 3866.7 (+41.6%) | 6940.8 (+154.3%) | 6145.8 | -11.4% |

### FP16 Training Speeds
| Model Name | RTX 3090 | RTX 4090 | RTX 5090 | RTX5090D | RTX5090D vs RTX5090 |
| --- | --- | --- | --- | --- | --- |
| VGG16 | 438.2 | 837.5 (+91.1%) | 1161.0 (+164.9%) | 1074.5 | -7.4% |
| ResNet50 | 888.7 | 1360.6 (+53.1%) | 1623.9 (+82.7%) | 1552.0 | -4.4% |
| TF EfficientNetV2 B0 | 1818.3 | 2823.5 (+55.3%) | 3446.1 (+89.5%) | 3299.7 | -4.2% |
| Swin Base Patch4 Window7 224 | 337.0 | 597.1 (+77.2%) | 822.3 (+144.1%) | 837.4 | +1.8% |
| EfficientViT M4 | 3114.9 | 3810.7 (+22.3%) | 7310.4 (+134.7%) | 6839.6 | -6.4% |

### FP32 Inference Speeds
| Model Name | RTX 5090 | RTX5090D | RTX5090D vs RTX5090 |
| --- | --- | --- | --- |
| VGG16 | 1867.5 | 2188.5 | +17.2% |
| ResNet50 | 3576.8 | 3578.1 | +0.0% |
| TF EfficientNetV2 B0 | 9254.5 | 8775.0 | -5.2% |
| Swin Base Patch4 Window7 224 | 1315.8 | 482.4 | -63.3% |
| EfficientViT M4 | 23555.6 | 21741.3 | -7.7% |

### FP16 Inference Speeds
| Model Name | RTX 5090 | RTX5090D | RTX5090D vs RTX5090 |
| --- | --- | --- | --- |
| VGG16 | 3350.1 | 3539.9 | +5.7% |
| ResNet50 | 5741.6 | 5605.3 | -2.4% |
| TF EfficientNetV2 B0 | 15907.3 | 14957.5 | -6.0% |
| Swin Base Patch4 Window7 224 | 2471.9 | 2562.6 | +3.7% |
| EfficientViT M4 | 31682.2 | 28326.9 | -10.6% |

### LLM Inference Speeds

| Model | RTX 5090 | RTX 5090D  | RTX5090D vs RTX5090 |
| --- | --- | --- | --- |
| deepseek-r1:32b | 60.66 | 58.45 | -3.64% |
| qwen2.5:32b | 62.81 | 58.53 | -6.81% |
| qwen2.5:7b | 213.48 | 191.18 | -10.44% |
| mistral-small:24b | 91.29 | 85.99 | -5.81% |
| phi4:14b | 130.31 | 118.73 | -8.89% |
| phi3.5:3.8b | 346.65 | 284.51 | -17.92% |
| llama3.1:8b | 210.79 | 182.93 | -13.22% |
| llama3.2:3b | 339.51 | 287.11 | -15.43% |
| qwen2.5:1.5b | 402.32 | 333.12 | -17.20% |

### Notes
- The "RTX5090D vs RTX5090" column shows the percentage increase or decrease of RTX5090D performance compared to RTX5090.

## Analysis

The comparison tables allow us to analyze the performance differences between RTX5090D and RTX5090. It seemed there are not much difference when run AI training and inferencing speed except **Swin Base Patch4 Window7 224 fp32**. 

And for LLM, when token rate is slow, there are not much difference too. But when token rate is fast, the difference is more obvious. 

I am not sure it is a software/environment issue or hardware difference due to I only have RTX5090D to test.

## How to Run the Benchmark

To reproduce our benchmark results, follow these steps:

1. Clone the repository containing the benchmarking script.
2. Install the required dependencies (`torch`, `timm`, `torchvision`, `tqdm`).
3. ollama is installed
4. Prepare the ImageNet dataset in the specified directory.
5. Run the scripts.

```bash
python benchmark_timm_models_train.py
python benchmark_timm_models_inference.py
python test_gpu_llm_token_rate.py
```

## File Descriptions

- benchmark_timm_models_train.py: This script is the core benchmarking tool. It measures the training speeds of various deep - learning models (such as VGG16, ResNet50, etc.) with different precisions (FP32 and FP16). Users can choose to use gpu caching for faster data loading or run the training without gpu caching. The script iterates over the training data, performs forward and backward passes, and records the time taken to calculate the training speed.

- benchmark_timm_models_inference.py: Focuses on benchmarking the inference speeds of models. It loads the validation data, moves a portion of it to the GPU, and measures how fast the models can generate predictions in both FP32 and FP16 precisions.

- test_gpu_llm_token_rate.py: using ollama to test the gpu token rate.

## Disclaimer
Please note that the benchmark results provided in this README are based on our testing environment and may vary depending on hardware configurations and other factors. We encourage users to conduct their own benchmarking to ensure accurate and up-to-date results.
For any questions or concerns, please feel free to contact us.

## Reference

https://nikolasent.github.io/hardware/deeplearning/benchmark/2025/02/17/RTX5090-Benchmark.html
