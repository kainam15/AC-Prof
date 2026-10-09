# 动态模块与自定义 Pipeline

[← 返回专题目录](runtime.md)

## Transformers dynamic-module 兼容生命周期
[`extensions/transformers`](../../acprof/extensions/transformers) 的每个精确版本维护两个 capability：

| 字段 | 原生 loader 必须满足的能力 |
| --- | --- |
| `local_dynamic_transitive_imports` | 空缓存、离线时递归准备 `A → B → C` 的本地相对依赖 |
| `local_dynamic_symlink_safe` | snapshot 文件指向 Hub blobs 时，仍按 snapshot 名称发现和加载相对依赖 |

4.57.6 和 5.6.0 的两项声明目前均为 `false`，继续使用
[`compat/transformers_dynamic.py`](../../acprof/container/compat/transformers_dynamic.py)。该 shim 保留已有行为：
复用 Transformers 的依赖发现、缓存路径和类导入，只补齐递归缓存文件，不修改 snapshot 或 blobs；
缺失源文件在执行入口前报告原 snapshot 路径，实际导入异常继续上传。
`false` 表示当前不能依靠完整的原生加载路径通过该项验收，不表示每个 symlink 场景均会失败。

只有两项都明确为 JSON 布尔值 `true`，公共入口才直接调用 Transformers
`get_class_from_dynamic_module()`；该分支不导入 shim，也不执行 AC-Prof 的依赖遍历或手工复制。
任一已评审能力为 `false` 时走 compat。未登记版本、目录版本不一致、字段缺失、`null` 或非布尔值
均 fail-closed，在导入自定义模型代码前报错；不使用 `version >= x`、最近版本或异常后回退来猜测。
版本别名及带本地构建后缀的版本也须独立登记。旧 `acprof.container.dynamic_modules` 入口已移除，
调用者统一使用 `acprof.container.local_pipeline`。

新增 runtime 的固定验收顺序为：**固定 tag/commit 上游源码确认 → transitive import test →
snapshot→blobs symlink test → offline + clean-cache test → 代表性模型 Interface Probe 与 Runtime Validation → 标记 capability=true**。
同时保留 circular import 终止、missing dependency 原路径诊断与 import exception 传播检查。
验收须保存精确包版本、源码 commit/SHA256、运行环境及测试/Probe 产物；不能仅凭新版号或某个 PR 已合并改表。

[`test_local_pipeline.py`](../../tests/test_local_pipeline.py) 验证两条路由和未知声明；
[`test_local_pipeline_native_runtime.py`](../../tests/test_local_pipeline_native_runtime.py) 复用已有五项依赖回归，
执行实际安装的上游 loader，同时禁止导入 shim。已登记 native runtime 自动执行该组；评审候选版本时，
在隔离的候选容器中设置仅供测试使用的 `ACPROF_TEST_NATIVE_TRANSFORMERS=<精确版本>`，并运行：

```bash
ACPROF_TEST_NATIVE_TRANSFORMERS=<精确版本> python scripts/run_tests.py \
  --pattern test_local_pipeline_native_runtime.py --require-no-skips \
  --report /evidence/native-dynamic-modules.json
```

测试只在子进程中临时替换 capability 声明，不替换上游 loader；实际安装版本不符会失败。
正式 loader 不读取这个测试开关。候选依赖需提前准备，执行回归时容器断网、snapshot 只读，
`HF_MODULES_CACHE` 指向新的可写临时目录。该组通过只证明动态加载契约，仍需在目标环境执行
Ultravox 等代表性 checkpoint 的 Interface Probe 与 Runtime Validation；不据此宣称 GPU、profiler 或完整模型推理已通过。
代表性 Probe 使用隔离候选 checkout／镜像中的临时能力声明，所有验收通过后才更新正式支持表。

shim 只有同时满足以下条件才能删除：

- 所有正式支持的 Transformers runtime 都原生支持 transitive relative imports。
- snapshot→blobs symlink、clean-cache + offline、circular 和 missing dependency 回归全部通过。
- 不再支持任何仍需 workaround 的旧 runtime。
- Ultravox 等代表性模型的 Interface Probe、Runtime Validation 及相关实际推理验证通过。

参考上游 [递归复制 PR #46022](https://github.com/huggingface/transformers/pull/46022)、
[symlink 修复讨论 PR #46611](https://github.com/huggingface/transformers/pull/46611) 和
[Transformers 5.13.0 loader 源码](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/dynamic_module_utils.py)
（Apache-2.0）。复用已维护的原生接口，不复制整套 loader，也不新增生产依赖；正式支持版本和锁不会因候选回归而升级。

## 本地模型声明与自定义 pipeline
`acprof run` 和 `acprof probe` 接受 `--model-spec /path/to/model.json`，覆盖 snapshot 中的
`acprof_model.json`。JSON 必须声明 `schema_version=1`、`format`、标准任务 `task`，最大 64 KiB。
制品接口使用 `format=onnxruntime/torchscript/skops` 和相对 snapshot 的 `model_file`；
ONNX 图像／文本还需[对应预处理声明](extensions.md#扩展声明与按需加载)。结构化任务的 `feature_dim`
可驱动输入生成，workload 中显式指定的不同宽度会报错。`--model-spec` 描述模型接口，
`--workload-spec` 描述输入样本、生成与实验参数，二者职责独立。

例如没有 Hub 任务标签的 Iris 可使用仓库中的声明与合成输入清单：

```bash
acprof run --model Ritual-Net/iris-classification \
  --model-spec examples/onnxruntime/iris.model.json \
  --workload-spec examples/onnxruntime/iris.json \
  --profiling-mode basic --cpus 1 --mems 2 --gpus off \
  --input-scales 1 --warmup 1 --repeat 2 --repeat-in-window 1 \
  --notify none --output-dir results/iris
```

[`iris.model.json`](../../examples/onnxruntime/iris.model.json) 选择 `iris.onnx`、
`tabular-classification`、4 列输入；无需修改上游仓库或添加 Iris 专用 handler。
这是运行链路示例，合成输入不用于 Iris 准确率评估。

声明自定义 pipeline 别名时，显式映射到已有任务协议，例如：

```json
{
  "schema_version": 1,
  "format": "transformers-pipeline",
  "task": "text-classification",
  "pipeline_task": "acme-classify"
}
```

`config.custom_pipelines` 必须包含所选 `pipeline_task` 及其 `impl`。已有 NLP、CV、Audio
pipeline handler 按该别名加载，输入生成、有效尺度和输出仍按标准 `task` 处理。
多模态文字输出可使用下述共享声明；其他自定义协议仍需 adapter。
`auto_map` 声明也只产生候选，不保证兼容锁定的 Transformers。
主机静态检查代码引用属于同一固定 snapshot，保存 `code_files/code_revision`；跨仓库代码引用
或缺少文件明确拒绝。自定义代码保留完整 snapshot，接口验证在无网络容器中执行；缺少依赖需
登记完整环境锁，不在验证或正式请求期间自动安装。

有效声明进入服务镜像构建指纹及 `runtime_environment.model_spec`，由构建期环境变量
`ACPROF_MODEL_SPEC_B64` 传入，server、独立验证与 profiler 使用同一份内容。不会改写 Hub
snapshot；本地文件路径与 SHA256 进入恢复身份，内容改变不能沿用旧实验。依赖层和模型文件层
仍可复用；改变下述离线依赖会重建模型文件层。发现与输出验证位于测量窗口外。
NLP/CV/Audio 自定义 pipeline 沿用各自 handler 的计时口径。

### 声明多模态输入与推理参数
`audio-text-to-text`、`image-text-to-text`、`video-text-to-text` 可通过 `multimodal` 声明
复用 `family-default` handler，无需按模型名称新增分支。以下是音频接口示例：

```json
{
  "schema_version": 1,
  "format": "transformers-pipeline",
  "task": "audio-text-to-text",
  "pipeline_task": "ultravox-pipeline",
  "multimodal": {
    "inputs": {"prompt": "text", "audio": "audio", "sampling_rate": "sampling_rate"},
    "forward_kwargs": {"max_new_tokens": "$max_new_tokens", "temperature": 0.0}
  }
}
```

`inputs` 的键是上游 `preprocess` 接收的字典字段，值引用 AC-Prof 解码后的输入：

| 标准任务 | 必须完整映射的输入 |
| --- | --- |
| `audio-text-to-text` | `text` 字符串、`audio` 单声道 NumPy 波形、`sampling_rate` |
| `image-text-to-text` | `text` 字符串、`image` RGB PIL 图像 |
| `video-text-to-text` | `text` 字符串、`video` 有序 RGB NumPy 帧、`fps` |

`forward_kwargs` 传入上游 `_forward`；`$max_new_tokens` 引用 workload 的生成上限，必须声明。
`$do_sample` 引用经过校验的 `false`；也可显式用 `do_sample=false` 或 `temperature=0`。
省略时默认映射 `max_new_tokens` 和 `do_sample`。不允许未知参数引用或随机采样声明。
当前协议仅支持 batch size 1、返回张量字典的 `preprocess` 和返回一个字符串的 `postprocess`；
必须保留 `input_ids` 及可识别的模态特征，缺失输入、尺度截断或输出不符会在独立验证时报错。

加载、媒体解码／上游 `preprocess`／设备传输、上游 `_forward`、输出解码分别落在现有四阶段，
正式推理窗口只调用 `_forward`。输出 token 数仍是对返回文字重新分词的计数；没有实际生成
token 证据时，相关生成速率／逐 token 工作量保持不可用，不用上限或字符数代替。

共享环境 `custom-multimodal-cpu/cu124/cu128` 在原 4.57.6 锁基础上加入 `peft==0.17.1`，
保留已有包版本。CPU 使用 FP32，GPU 使用 FP16。参考
[Transformers Pipeline 契约](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/base.py)
和 [UltravoxPipeline](https://github.com/fixie-ai/ultravox/blob/main/ultravox/model/ultravox_pipeline.py)
拆分执行阶段；前者与 PEFT 为 Apache-2.0，Ultravox 代码为 MIT。模型代码来自固定 snapshot，
只在容器中执行。其他依赖组合仍需登记独立完整环境锁。

### 外部模型与 processor 的离线依赖
可选 `dependencies` 数组声明上游代码在加载时还会读取的 Hub 仓库，每项包含 `repo_id`、
固定 40 位 commit `revision`，以及可选的相对文件 `allow_patterns`。最多 16 个不同仓库，
不能覆盖主模型；不声明 patterns 时下载依赖仓库完整 snapshot。主机准备阶段校验文件哈希后，
在 Model Store 的固定计划视图中将各依赖的 `refs/main` 绑定到声明 commit，容器只读挂载该视图，
使上游无 revision 的 `from_pretrained(repo_id)` 也能离线解析。主模型和全部依赖进入下载计划与镜像身份，正式服务保持断网。

[Ultravox 完整声明](../../examples/multimodal/ultravox.model.json) 包含固定版本的 Llama 基础权重与
Whisper processor。Llama 仓库要求账号已获访问许可，并在主机准备阶段配置有效 `HF_TOKEN`；
缺少权限不能靠修改任务覆盖项解决。在 TUI 的“高级参数 → 识别覆盖 → 模型接口声明”填入
`examples/multimodal/ultravox.model.json`，或在 CLI 使用：

```bash
acprof run --model fixie-ai/ultravox-v0_5-llama-3_2-1b \
  --model-spec examples/multimodal/ultravox.model.json \
  --profiling-mode basic --cpus 2 --mems 12 --gpus on \
  --input-scales 1 --batch-size 1 --warmup 1 --repeat 2 \
  --repeat-in-window 1 --notify none --output-dir results/ultravox-basic
```

此命令是配置示例，不表示完整 checkpoint 已验证通过；声明后的状态仍为 `candidate`，
加载、预处理、推理和输出验证均成功后才成为本次运行的 `verified`。
