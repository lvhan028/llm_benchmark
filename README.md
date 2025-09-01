# LMDeploy Benchmark

该项目的主要目的是提供 LMDeploy 里程碑版本在 H800、A100 等硬件设备上的推理性能基准，为用户提供选型参考和性能预期。此外，项目也会不定时提供 vLLM、SGLang、TensorRT-LLM 等项目的测试结果，为用户提供多框架性能对比和优化方向参考。

## 基准测试设置

我们参考了多方的测试数据，设计了如下4种测试场景：

1. ShareGPT 短输入输出

   ShareGPT 已成为各推理框架公认的测试数据集

2. 固定输入、输出各 1k token

   我们查阅了 PyTorch CI HUD vLLM benchmark、SGLang benchmark、TensorRT-LLM benchmark，发现都有这组参数下的测试结果。所以，把它加入了测试基准。

3. 长思考场景

   随着模型思考能力提升，大量算力消耗在 token 生成阶段。DeepSeek在报告中提及，“平均每输出一个 token 的 KVCache 长度是 4989”，接近 5000 token。我们使用输入 2000 token、输出 6000 token 来简化模拟这种场景。

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

我们选择了Qwen3、gpt-oss 等热门模型，在 H800、A100 硬件平台上对 LMDeploy 的推理性能进行了全面测试。详细测试结果请参考一下链接

### H800

- [LMDeploy-v0.9.2](./lmdeploy/H800_v0.9.2.md)
- [LMDeploy-v0.10.0](./lmdeploy/H800_v0.10.0.md)

### A100

- [LMDeploy-v0.9.2](./lmdeploy/A800_v0.9.2.md)
- [LMDeploy-v0.10.0](./lmdeploy/A100_v0.10.0.md)
