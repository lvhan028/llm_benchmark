# LMDeploy Benchmark

该项目的主要目的是提供 LMDeploy 里程碑版本在 H800、A100 等硬件设备上的推理性能基准，为用户提供选型参考和性能预期。此外，项目也会不定时提供 vLLM、SGLang、TensorRT-LLM 等项目的测试结果，为用户提供多框架性能对比和优化方向参考。

## 基准测试设置

我们参考了多方的测试数据，设计了如下4种测试场景：

1. ShareGPT 短输入输出

   ShareGPT 已成为各推理框架公认的测试数据集

2. 固定输入、输出各 1k token

   我们查阅了 [PyTorch CI HUD vLLM benchmark](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm)、[SGLang benchmark](https://github.com/sgl-project/sglang/blob/main/benchmark)、[TensorRT-LLM benchmark](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-overview.md)，发现都有这组参数下的测试结果。所以，把它加入了测试基准。

3. 长思考场景

   随着模型思考能力提升，大量算力消耗在 token 生成阶段。DeepSeek在[报告](https://zhuanlan.zhihu.com/p/27181462601)中提及，“平均每输出一个 token 的 KVCache 长度是 4989”，接近 5000 token。我们使用输入 2000 token、输出 6000 token 来简化模拟这种场景。

4. 长输入场景

   调换了“长思考场景”中的输入和输出，使用输入 6000 token、输出 2000 token 表示该场景

以上场景，我们定义在了测试脚本 `benchamrk_serving.sh` 中：

```shell
configs=(
    "--dataset-name sharegpt --num-prompts 10000"
    "--dataset-name random --num-prompts 2000 --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0"
    "--dataset-name random --num-prompts 500 --random-input-len 2000 --random-output-len 6000 --random-range-ratio 1.0"
    "--dataset-name random --num-prompts 500 --random-input-len 6000 --random-output-len 2000 --random-range-ratio 1.0"
)
```

## 基准测试指标

- **RPS/QPS (req/s)**：每秒处理的请求数量（Requests Per Second / Queries Per Second）

- **Token 吞吐量 (token/s)**：每秒中生成的 token 的数量

- **TPOT (ms)**：Time Per Output Token，生成一个 token 所耗时间（单位为毫秒）

- **ITL (ms)**：Inter-Token Latency，指从上一次收到 token 到当前收到 token 之间的时间间隔（单位为毫秒）

## 基准测试方法

基准测试过程分为模型部署和性能测试两个阶段。以模型 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) 为例，具体操作步骤如下：

1. 模型服务部署

首先使用以下命令将模型部署为推理服务：

```shell
lmdeploy serve api_server Qwen/Qwen3-8B
```

2. 性能测试执行

待服务启动后，运行基准测试脚本以获取 LMDeploy 框架下 Qwen3-8B 模型的推理吞吐性能：

```shell
bash benchmark_serving.sh --backend lmdeploy
```

该测试流程将输出包括吞吐量、延迟等关键性能指标，为模型部署提供量化参考依据。

## 基准测试结果

我们选择了Qwen3、gpt-oss 等热门模型，在 H800、A100 硬件平台上对 LMDeploy 的推理性能进行了全面测试。详细测试结果请参考以下链接

### H800

- [LMDeploy-v0.9.2](./lmdeploy/H800_v0.9.2.md)
- [LMDeploy-v0.10.0](./lmdeploy/H800_v0.10.0.md)

### A100

- [LMDeploy-v0.9.2](./lmdeploy/A800_v0.9.2.md)
- [LMDeploy-v0.10.0](./lmdeploy/A100_v0.10.0.md)

## 推理参数调优

在模型并行方式确定之后，影响 LMDeploy 推理性能的主要参数有：

- --cache-max-entry-count

  该参数表示 k/v cache 占用空闲显存的比例，默认值为 0.8。空闲显存表示加载完模型权重后剩余的显存。如果遇到 OOM，需酌情调低该比例值，降低 k/v cache 的显存占用量

- --max-batch-size

  该参数定义了在解码（decoding）阶段，单次前向推理中能够处理的最大请求批处理数量。一般来说，max_batch_size 越大，token 吞吐量越高，但 token 延迟也会相应增加。反之，吞吐量降低，token 延迟缩短。

  LMDeploy 综合考虑吞吐和延时之间的平衡，为不同型号的显卡设定了以下默认值：

  - **v0.10.0 之前**，A100/A800 为 256，H100/H800 为 512，其他显卡均为 128。
  - **v0.10.0 及之后**，A100/A800 为 512，H20/H100/H800 为 1024，H200 为 1280，其他显卡均为 128。

  ```python
  # before v0.10.0
  import torch
  device_name = torch.cuda.get_device_name(0).lower()

  max_batch_size_map = {'a100': 256, 'a800': 256, 'h100': 512, 'h800': 512}
  for name, size in max_batch_size_map.items():
     if name in device_name:
           return size
  return 128

  # since v0.10.0
  max_batch_size_map = {'a100': 512, 'a800': 512, 'h100': 1024, 'h800': 1024, 'h200': 1280, 'h20': 1024}
  for name, size in max_batch_size_map.items():
     if name in device_name:
           return size
  return 128
  ```

  ```{note}
  某些显卡（尤其是特供版本）的 device_name 可能无法被正确映射到 LMDeploy 定义的标准。在这种情况下，建议用户通过 `--max-batch-size` 参数手动指定该值，以充分发挥 LMDeploy 推理性能。
  ```
