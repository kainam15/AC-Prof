# 任务接口与支持范围

[← 返回专题目录](runtime.md)

## 任务支持范围

目前支持的任务族：

| 任务族 | `input_scale` 的含义 | 示例 |
| --- | --- | --- |
| NLP | token 序列长度；表格问答为行数 | BERT、文本生成、问答、句子相似度、文本排序 |
| CV | 基础图像／视频帧尺寸的缩放倍率 | 图像分类、目标检测、图像描述、关键点、视频分类 |
| Audio | 输入音频秒数；文本生成音频为输入 token 数 | ASR、分类、语音／音频生成、codec 重建、VAD |
| Time series | context length | Chronos 时间序列预测 |
| Diffusion | 图像／视频帧边长；无条件图像和 3D 为去噪步数 | 文生图、图像编辑、视频生成、Shap-E 网格生成 |
| Multimodal | 依任务为输入图像边长、音频秒数或视频帧数 | 多模态问答、文档检索、文字＋音频输出 |
| Structured | 表格行数、独立观测数或每图节点数 | 表格分类／回归、离线策略推理、图模型 |

大多数 Hugging Face 模型会自动识别任务族和后端；识别失败时再使用 `--task`、`--task-family` 或 `--backend` 覆盖。没有任务标签的 Diffusers 模型可根据固定 revision 的 `model_index.json` 中已登记的原生 pipeline 类名识别。CV 每个请求使用一张图或一个视频，要求 `--batch-size 1`。未登记的任务类型、任务族／后端不匹配及不支持的 batch 会被提前拦截；通过预检仍需模型架构和容器依赖兼容。

图像描述的输出字段、token 计数与尺度边界见[图像描述契约](#图像描述输出与兼容范围)。

CV 镜像同时安装 `build-essential`，供 PyTorch/Triton 在首次 GPU 推理时编译所需模块，以及 VitPose 图像变换需要的 SciPy。首次验证 BLIP 可运行以下命令。适配代码或依赖变化后需构建匹配指纹的镜像；复用前核验环境清单。

```bash
acprof run --model Salesforce/blip-image-captioning-base \
  --cpus 2 --mems 8 --gpus off,on --input-scales 1 --batch-size 1 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none \
  --notify none --output-dir results/smoke-blip
```

`text-to-image` 模型会自动选择 `diffusion` 任务族和 `diffusers` 后端。内置 workload 固定提示词、随机种子、guidance scale 和 20 个去噪步，只改变输出分辨率；服务端仅返回生成图像的数量与尺寸元数据，避免图片响应体影响网络和应用延迟测量。

### NLP、音频、表格和策略任务

以下 24 个任务类别接入统一输入计划与采集流程。适配范围以表中模型接口／文件格式为准，同一个 Hub 标签可能包含多种不兼容的框架。适配代码或依赖变化后需使用匹配指纹的镜像，构建与复用按[上述契约](images.md#构建复用和验证)执行。

| Hugging Face 任务 | 适配接口／格式 | 输入尺度 |
| --- | --- | --- |
| `text-classification` | Transformers 分类 pipeline | 文本 token 数 |
| `token-classification` | Transformers token 分类 pipeline | 文本 token 数 |
| `table-question-answering` | Transformers 表格问答 pipeline；列数组表格＋query | 表格行数，默认 1、2、4、8、16、32 |
| `question-answering` | Transformers 问答 pipeline；question＋context | context token 数，问题固定 |
| `zero-shot-classification` | Transformers NLI pipeline；固定候选标签和假设模板 | 输入文本 token 数 |
| `translation` | Transformers 翻译 pipeline | 输入 token 数 |
| `summarization` | Transformers 摘要 pipeline | 输入 token 数 |
| `feature-extraction` | SentenceTransformer 完整模块图的句向量，或普通 Transformers token 特征 pipeline；依据 library／`modules.json` 选择 | 输入正文 token 数，prompt 另占模型容量 |
| `text-generation` | Transformers 生成 pipeline | 输入 token 数 |
| `fill-mask` | Transformers 掩码填充 pipeline，自动使用 tokenizer 的 mask token | 输入 token 数 |
| `sentence-similarity` | SentenceTransformer 编码并计算相似度；后端 `sentence_transformers` | 候选文本 token 数的最大值，query 和候选数量固定 |
| `text-ranking` | CrossEncoder 成对评分；后端 `cross_encoder` | 候选文本 token 数的最大值，query 和候选数量固定 |
| `text-to-speech` | Transformers TextToAudioPipeline 的自包含波形模型，如 VITS／Bark | 输入文本 token 数 |
| `text-to-audio` | 同一官方 pipeline 的自包含文本条件模型，如 MusicGen | 输入文本 token 数 |
| `automatic-speech-recognition` | Transformers ASR pipeline | 输入音频秒数 |
| `audio-to-audio` | Transformers Encodec／DAC 音频编码后重建 | 输入音频秒数 |
| `audio-classification` | Transformers 音频分类 pipeline | 输入音频秒数 |
| `voice-activity-detection` | Silero `silero_vad.jit`；后端 `torchscript`，仅 CPU | 输入音频秒数 |
| `tabular-classification` | skops 保存的 sklearn 分类器，或约定格式的 TorchScript | 每个 batch 项的表格行数 |
| `tabular-regression` | skops 保存的 sklearn 回归器，或约定格式的 TorchScript | 每个 batch 项的表格行数 |
| `time-series-forecasting` | `BaseChronosPipeline` 按配置分派 Chronos／Chronos-Bolt／Chronos-2；单变量序列列表 | 历史时间步数 |
| `reinforcement-learning` | TorchScript 向量观测策略 | 每个 batch 项的独立观测数 |
| `robotics` | TorchScript 向量观测策略 | 每个 batch 项的独立观测数 |
| `graph-ml` | TorchScript `forward(x, edge_index, batch)` | 每张图的节点数 |

NLP 的输入计划保存真实 payload，句子相似度／排序每次重新编码 query 和文档，零样本分类完整运行候选标签对应的 NLI 推理。表格问答固定列结构并改变行数，超出模型容量时失败，不通过删行伪装成原尺度。音频任务要求 `--batch-size 1`；读取音频的任务默认复用有来源与 SHA256 的内置 LibriSpeech 前缀，文本到音频使用确定性文本。生成音频仅返回形状、采样率、样本数和时长摘要。需要额外声码器／说话人资产的 SpeechT5、FastSpeech2Conformer 暂未适配，会在加载时明确拒绝。

SentenceTransformer 保留仓库的 Pooling、Normalize 和默认 prompt；特征提取输出为 `[batch, embedding_dim]`
摘要，不再把 token 隐状态当作句向量。可通过 [NLP workload 参数](../usage/cli-run.md#nlp-workload-参数)
选择 `prompt`／`prompt_name` 和 `normalize_embeddings`，输入预算包含 prompt。句子相似度仍为对称
`encode`，没有增加独立 `encode_query`／`encode_document` 协议。Chronos 三代共用单变量列表输入；
原生 list 输出核对 batch、变量数和 horizon 后合并为 `[batch, samples_or_quantiles, horizon]`，不把不同代的
sample 和 quantile 数值语义混为一谈。输入序列留在 CPU，由原生 pipeline 批处理、固定内存及搬运到
模型设备，这些工作包含在 `predict` 与对应 profiler 范围内；不提前传入 CUDA tensor 破坏 DataLoader 约定。

例如运行一个表格问答尺度，或将 `--task` 换为表中任务并选择对应模型：

```bash
acprof run --model google/tapas-base-finetuned-wtq \
  --task table-question-answering --cpus 2 --mems 8 --gpus off \
  --input-scales 4 --batch-size 1 --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none --notify none \
  --output-dir results/smoke-table-qa
```

表格分类／回归、强化学习、机器人和图任务使用 `structured` 任务族。TorchScript 模型仓库需同时提供模型文件和以下 `acprof_model.json`；`task` 必须匹配，`feature_dim` 必须等于输入宽度。表格／策略的模型签名为 `forward(features)`／`forward(observations)`，输入为 FP32 `[batch_size × input_scale, feature_dim]`；图模型接收 FP32 节点特征、INT64 COO 边与 INT64 图编号，按不相连图合批。输出要求一个至少一维的 tensor／array。

```json
{"schema_version": 1, "task": "robotics", "format": "torchscript", "model_file": "model.pt", "feature_dim": 7}
```

`--workload-spec` 指定结构化输入宽度、尺度和种子，例如：

```json
{"schema_version": 1, "task": "robotics", "feature_dim": 7, "input_scales": [1, 8, 32, 128], "seed": 12345}
```

默认表格宽度 8、强化学习 4、机器人 7、图节点特征 16；应按模型更改。非图默认尺度 1、8、32、128，图为 8、32、128、512；固定特征宽度，图结构为双向环。skops 使用 `--backend skops --gpus off`，仓库内唯一 `.skops` 可自动发现，也可由模型清单指定；仅加载 sklearn 已知类型，模型版本须与镜像的 sklearn 版本兼容。

策略任务测量独立向量观测的前向推理，不包含环境交互、训练、回报评估、传感器采集和机器人执行；图像／多模态策略需要另行适配输入，不能通过展平图像宣称等价兼容。结构化合成输入用于性能流程，不能据此报告模型准确率或策略效果。TorchScript 是已被上游标记 deprecated 的导出兼容接口，兼容性取决于导出版本及算子；不能在加载时更换注意力实现，`torch_profiler_eager` 因此会明确拒绝。其它 profiler 仍按各自适用条件隔离执行。可运行的五类导出示例见 [export_models.py](../../examples/structured/export_models.py)，对应真实四阶段检查见 [smoke.py](../../examples/structured/smoke.py)。

实现复用 [Transformers 4.57.6 pipeline](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/__init__.py)、[Sentence Transformers 5.1.2](https://github.com/huggingface/sentence-transformers/tree/v5.1.2)、[PyTorch 模型加载](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/jit/_serialization.py) 和 [skops](https://github.com/skops-dev/skops)。Transformers／Sentence Transformers 为 Apache-2.0，PyTorch 为 BSD 风格许可，skops 为 MIT；模型权重另按其许可。NLP 镜像补齐 pandas 与编码器依赖，结构化镜像隔离安装 sklearn／skops，主机不安装推理框架。LeRobot 的运行时依赖与本项目 Python 3.10 镜像不同，旧 Graphormer 位于 Transformers 的 deprecated 目录，因此策略／图任务采用显式导出接口；没有引入模拟器或旧框架。素材生成、哈希与输入规划在正式测量窗口之外执行。

运行接口验证可使用 [NLP 十二任务示例](../../examples/nlp/smoke.py)、[音频原生模型测试](../../tests/test_audio_runtime_optional.py) 和 [Chronos 小模型示例](../../examples/structured/chronos_smoke.py)。这些脚本需要对应容器的依赖，使用随机小模型／确定性导出样例验证输入和推理接口，不提供真实模型准确率或性能结论。文本到音频目前使用内置文本，不能传 WAV 清单作为 `--workload-spec`。

### 视觉任务

以下 19 类任务接入同一输入计划、最大输入探测、采集和后置 profiler 流程。适配以镜像中 Transformers 4.57.6／Diffusers 0.39.0 的原生接口为边界，不表示 Hub 上同标签的任意模型或自定义代码均可运行。

timm 图像分类使用 Transformers 官方 `timm_wrapper` 与已有 CV handler；从 `pretrained_cfg` 读取
预处理配置，CV 环境提供锁定 timm。更换兼容 timm checkpoint 无需新增模型 ID 分支。
需要较新原生架构时按[共享接口解析](routing.md#共享接口解析)选择 5.6.0 候选环境。

| Hugging Face 任务 | 任务族 | 适配范围 |
| --- | --- | --- |
| `depth-estimation` | CV | 原生深度估计 pipeline，返回深度图摘要 |
| `image-classification` | CV | 原生图像分类 pipeline |
| `object-detection` | CV | 原生目标检测 pipeline |
| `image-segmentation` | CV | 原生语义／实例／全景分割 pipeline |
| `text-to-image` | Diffusion | 原生文本条件图像生成 |
| `image-to-text` | CV | 原生图像描述 pipeline，返回文字及 token 数 |
| `image-to-image` | Diffusion | 原生 Img2Img／图像编辑／图像变体；须满足当前方形输出尺度约定 |
| `image-to-video` | Diffusion | 原生图生视频，包括无文本条件的 Stable Video Diffusion |
| `unconditional-image-generation` | Diffusion | DDPM／DDIM 等原生无条件生成；模型固定输出尺寸，扫描去噪步数 |
| `video-classification` | CV | `AutoModelForVideoClassification` 与图像处理器；固定帧数，扫描帧分辨率 |
| `text-to-video` | Diffusion | 原生文本条件视频生成 |
| `zero-shot-image-classification` | CV | 原生 pipeline，同时传入候选标签 |
| `mask-generation` | CV | 原生 SAM 自动掩码 pipeline |
| `zero-shot-object-detection` | CV | 原生零样本检测 pipeline，同时传入候选标签 |
| `text-to-3d` | Diffusion | Shap-E 文本条件网格生成，扫描去噪步数 |
| `image-to-3d` | Diffusion | Shap-E 图像条件网格生成，扫描去噪步数 |
| `image-feature-extraction` | CV | 原生图像特征 pipeline，返回特征形状摘要 |
| `keypoint-detection` | CV | SuperPoint 关键点、VitPose／VitPose++ 姿态估计 |
| `video-to-video` | Diffusion | 原生视频条件生成，消费有序输入帧 |

除 `text-to-image` 外，上表任务均要求 `--batch-size 1`。CV 的 `input_scale=1` 仍表示 224×224 输入图像或帧，视频默认 16 帧。模型要求的帧数必须与清单一致，不静默丢帧或补帧；图像 processor 可能缩放输入，输入像素大小不能直接视为模型内部计算规模。

CV 可使用 `--workload-spec` 指定图片、视频帧、候选标签、姿态框和推理参数。图片字段为 `image_path`，视频为有序的 `video_frames` 路径列表；路径相对清单文件解析。未提供素材时使用确定性合成图／帧，零样本默认标签为 `cat,dog,car,person`。例如：

```json
{
  "schema_version": 1,
  "input_scales": [0.5, 1.0, 2.0],
  "candidate_labels": ["cat", "person"],
  "params": {"threshold": 0.2}
}
```

该示例用于 `zero-shot-object-detection`；视频可用 `num_frames` 设置合成帧数，也可由 `video_frames` 数量确定。VitPose 清单的 `boxes` 是归一化到 0–1 的 COCO `[x,y,width,height]`，默认全图框；主机按输入尺寸转换为像素坐标，不额外运行人物检测器。VitPose++ 可在 `params` 中指定 `dataset_index`。这些合成输入用于性能流程验证，不是准确率评测集。

无条件图像与 3D 使用 `input_scale_type="denoising_steps"`，默认尺度 `1,2,4,8,16,20`；通过 `--input-scales` 或清单 `input_scales` 修改，不再同时用 `params.num_inference_steps` 指定。DDPM 分辨率来自模型，Shap-E 直接解码网格，不用渲染图像的 `frame_size` 冒充 3D 工作量。网格仅返回数量、顶点和面数摘要，不传输网格文件。其它生成任务继续扫描方形输出边长，默认视频 17 帧；模型必须满足对应分辨率、帧数与条件参数约束。

`image-to-image`／`image-to-video` 的合成默认提示词仅在原生接口接受 `prompt` 时使用；显式写入清单的提示词必须被消费，不接受文字条件的模型会拒绝该清单。Diffusers 图像到图像可通过官方 `AutoPipelineForImage2Image.from_pipe` 复用已有组件；视频到视频仅转换已适配的 CogVideoX 文生视频、单 denoiser Wan 和旧版 TextToVideoSD pipeline，无法保留双 denoiser 等组件时明确失败。固定倍率的 Stable Diffusion Upscale／LatentUpscale 暂不符合当前方形输出边长约定，会明确拒绝。旧版 TextToVideoSD／VideoToVideoSD 上游已停止更新，保留固定版本兼容；现代视频模型另按原生参数检查。

实现复用 [Transformers 原生 pipeline](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/pipelines/__init__.py)、[VitPose 接口](https://github.com/huggingface/transformers/blob/v4.57.6/docs/source/en/model_doc/vitpose.md)、[Diffusers DDPM](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/ddpm/pipeline_ddpm.py) 与 [Shap-E](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/shap_e/pipeline_shap_e.py)。两个上游采用 Apache-2.0、仍持续维护；沿用已固定版本，仅为姿态处理新增 SciPy。视频采用主机预先准备的 PNG 帧，不引入视频编解码库，素材生成和哈希计算均在测量窗口之前完成。

### 多模态任务

下列 9 类任务已接入任务识别、输入计划、容器处理器、最大输入探测、正式采集与后置 profiler。新任务统一要求 `--batch-size 1`。它们使用已安装版本中的原生模型接口；支持任务类型不表示任意同标签 checkpoint 都兼容。

| Hugging Face 任务 | 后端 / 任务族 | 适配范围与默认输入尺度 |
| --- | --- | --- |
| `audio-text-to-text` | Transformers / `multimodal` | 原生 Auto 文本生成模型及其音频 chat processor，如 Voxtral、Qwen2 Audio、Qwen2.5 Omni Thinker；或声明多模态接口的 custom pipeline；MOSS 保留独立 adapter；真实语音＋文字；1、2、5、10 秒 |
| `image-text-to-text` | Transformers / `multimodal` | `AutoModelForImageTextToText` 支持且带 chat template 的原生模型，或声明多模态接口的 custom pipeline；224、336、448 像素输入边长 |
| `image-text-to-image` | Diffusers / `diffusion` | 原生同时接收 `image` 和 `prompt` 的图像编辑／Img2Img pipeline；128–512 像素输出边长 |
| `image-text-to-video` | Diffusers / `diffusion` | 原生同时接收图像和文本的 CogVideoX、Wan 等 I2V pipeline；方形帧，默认固定 17 帧 |
| `visual-question-answering` | Transformers / `multimodal` | 原生 VQA pipeline，区分分类式与生成式回答；224、336、448 像素 |
| `document-question-answering` | Transformers / `multimodal` | 原生 DocQA pipeline；内置可读票据与词框；外部文档须给出 OCR 词和坐标 |
| `video-text-to-text` | Transformers / `multimodal` | 同时支持视频 processor 和图文生成 Auto 类的模型，或声明多模态接口的 custom pipeline；2、4、8 帧，固定 2 FPS |
| `visual-document-retrieval` | Transformers / `multimodal` | ColPali、ColQwen2；每次编码一个 query 和一页文档，再计算 MaxSim 分数 |
| `any-to-any` | Transformers / `multimodal` | Qwen2.5 Omni 的文字／图像／音频／视频输入 → 文字＋音频输出；默认输入为语音＋文字 |

共享音频输入先校验单声道 PCM WAV、采样率和处理器长度上限，再把音频与文字一起交给
`processor.apply_chat_template(tokenize=True, return_dict=True)`。音频采用临时本地 WAV 消息，
可同时供通用 `ProcessorMixin` 和原生 tokenizer 消费；临时文件在预处理结束或异常时删除，
`predict` 只复用张量执行 `generate`。模板及 processor 参数按方法公开签名传递，
不要求所有原生 processor 都有 Jinja `chat_template`，不把已生成的音频模板再当纯文本编码。
返回值必须包含 `input_ids` 和非空音频特征；输出继续通过既有文本生成证据与 `validate_output` 验证。

实现参考官方 [Auto 注册表](https://github.com/huggingface/transformers/blob/v5.6.0/src/transformers/models/auto/modeling_auto.py)、
[Voxtral processor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/voxtral/processing_voxtral.py)
和 [mistral-common](https://github.com/mistralai/mistral-common)。沿用上游 Apache-2.0 的原生接口，
不引入另一套推理引擎；在 4.57.6 多模态和 5.6.0 共享环境固定 `mistral-common[audio]==1.11.7`，
其传递依赖进入 CPU/cu124/cu128 的精确锁，已有包不随之升级。额外 WAV 物化和特征提取计入
预处理，不进入仅推理计时。架构能加载仍不表示设备容量足够，例如 Voxtral-24B 的完整权重
不适合在 8 GB 显存上按当前半精度默认设置运行；量化与 offload 不是本次新增能力。

实际 Hub 标签 `image-to-image`、`image-to-video` 也会接入 Diffusers 适配。显式的 `image-text-to-image`／`image-text-to-video` 任务要求模型同时接收文字和图像条件；`image-to-video` 也可使用无文本的原生 pipeline。模型如需要非方形输出、更大的分辨率、不同帧数或额外组件，需要满足其自身约束；仅采用上述官方原生 pipeline 转换，不执行 Diffusers 自定义远程代码。

TUI 的高级配置可选 `Multimodal`，也可使用 CLI。首次运行应重建模型镜像，后续再用 `--skip-build` 复用。以下例子只运行一个 VQA 输入尺度：

```bash
acprof run --model dandelin/vilt-b32-finetuned-vqa \
  --task visual-question-answering --task-family multimodal \
  --backend transformers_model --cpus 2 --mems 8 --gpus off \
  --input-scales 224 --batch-size 1 --warmup 0 --repeat 1 \
  --repeat-in-window 1 --compute-profile-tool none \
  --execution-profile-tool none --notify none --output-dir results/smoke-vqa
```

图像和视频默认使用确定性的合成场景；DocQA 使用带词框的合成票据，音频复用仓库内带来源与 SHA256 的真实语音。它们适合验证采集与性能流程，不是准确率评测集。图像 processor 可能缩放或切块，因此输入像素边长不等于模型实际视觉 token 数。

`--workload-spec` 可为新任务指定本地 JSON。多模态清单使用 `text`，Diffusers 清单使用 `prompt`；文件路径相对清单所在目录。图片示例：

```json
{
  "schema_version": 1,
  "task": "image-text-to-text",
  "image_path": "scene.png",
  "text": "Describe this image briefly.",
  "input_scales": [224, 448],
  "params": {"max_new_tokens": 32, "do_sample": false}
}
```

多模态清单还支持 `audio_path`（单声道 PCM16 WAV，采样率须符合模型）、`video_frames`（有序本地图片路径列表）、`fps`、`image_resolution`（视频帧／固定条件图边长）。视频在主机侧准备成 PNG 帧，音频使用同一波形的前缀，主采集与 profiler 复用计划内的 Base64 素材，不在容器运行时下载。自定义文档需同时提供 `words` 和归一化到 0–1000 的 `boxes`，避免运行 OCR；仍需模型所需的图像后端依赖，依赖 detectron2 等额外组件的模型不包含在镜像默认支持范围内。

Any-to-Any 可用 `"modalities": ["image", "audio"]` 和 `"scale_modality": "audio"` 组合输入；一次只改变一个维度，其余由 `image_resolution`、`audio_duration_s`、`video_num_frames` 固定。音频生成保持开启（`return_audio=true`），默认文本最多 64 token、talker 最多 256 token、`speaker="Chelsie"`、关闭 token 采样，并用 `seed=12345` 固定声码器噪声。固定种子不保证跨硬件／软件版本逐位一致。这里的 Any-to-Any 明确为 Omni 的文字＋音频输出，未实现任意图像／视频输出协议。

Diffusers 清单示例（还可设置 `strength`、`image_guidance_scale`、`negative_prompt`，目标 pipeline 必须支持所传参数）：

```json
{
  "schema_version": 1,
  "image_path": "scene.png",
  "prompt": "The camera slowly moves to the left.",
  "input_scales": [256, 512],
  "params": {"num_inference_steps": 20, "num_frames": 17, "guidance_scale": 7.5, "seed": 12345}
}
```

清单及素材摘要、实际 payload、参数和尺度单位保存在 `input_scale_plan.json` / `static_meta.json`。检索请求包含 query 编码、文档编码和评分，不缓存文档向量。生成图像、视频、音频只返回尺寸／数量摘要，不编码为响应媒体；文字输出按既有 CSV 字段统计，检索分数不会冒充输出 token。详细单位见[输入规模与音频清单](../usage/cli-run.md#输入规模与音频清单)。

普通采集与 NCU／Nsys 复用完整 `predict()`。profiler 在推理计算捕获前的预热阶段验证一次输出协议，计算捕获只重复推理；Massif 按整个进程生命周期统计，包含加载、预热和这次验证。Omni 的 Token2Wav 不支持 eager 注意力，因此 `any-to-any` 的 `torch_profiler_eager` 会明确失败；Diffusers 的 Transformer 视频模型也会拒绝尚未验证的 eager 替换。这些失败按工具隔离，不能把未采集的 FLOP 当成 0。已有 UNet 文生图 eager 路径保留。

适配复用官方 [Transformers 多模态接口](https://github.com/huggingface/transformers/blob/v4.57.6/docs/source/en/chat_templating_multimodal.md)、[检索接口](https://github.com/huggingface/transformers/blob/v4.57.6/docs/source/en/tasks/visual_document_retrieval.md)、[Omni 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py) 和 [Diffusers pipeline](https://github.com/huggingface/diffusers/tree/v0.39.0/src/diffusers/pipelines)。原生多模态路径默认使用 Transformers 4.57.6；需要新版 Auto 类时按元数据选择共享的 5.6.0 环境，仍受现有任务协议限制。Diffusers 保持 0.39.0，MOSS 保留独立的 Transformers 5.6.0 环境。两库采用 Apache-2.0；具体模型权重的许可与访问条件以其模型页为准。

MOSS 自动选择专用 adapter 和依赖锁，不需要修改主机 `.venv`。默认提示词要求带时间戳和说话人编号的转写，`max_new_tokens=512`，CPU 使用 FP32，GPU 使用 BF16；processor 按官方方式分块处理音频，输出保留原始标记文本。可先运行：

```bash
acprof run --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --cpus 1 --mems 8 --gpus off,on --input-scales 1 --batch-size 1 \
  --warmup 0 --repeat 1 --repeat-in-window 1 \
  --compute-profile-tool none --execution-profile-tool none \
  --notify none --output-dir results/smoke-moss
```

8 GiB 是上述冒烟测试的容器内存配置，不是模型的最低内存保证。不同依赖版本的选择、验证与新模型接入方法见本页的[构建复用和验证](images.md#构建复用和验证)。

### 图像描述输出与兼容范围

- 默认 CV profile 使用 Transformers 4.57.6 的官方 `image-to-text` pipeline；最终 profile 的架构与 task registry 必须同时满足。5.6.0 缺少该 pipeline 时在预检返回 `runtime_task_unsupported`。镜像实际版本不符时按所选 lock 重建；新的代码和依赖不会自动写入已存在的 Docker 镜像。
- `/predict` 返回 `task="image-to-text"`、`output_type="caption"`、`captions: string[]`、`n_results`、`output_length` 和可空的 `output_token_count`。一次请求输入一张图；若生成多条候选，`n_results` 为候选数，字符/token 指标为该请求所有候选之和。空字符串是有效输出，缺少 `generated_text`、非字符串内容或没有候选则报请求错误，不计为成功检测结果。
- 输入 `params` 直接传给官方 pipeline，缺省时使用该 pipeline 与固定模型 revision 的默认生成配置。响应文本解析与重新分词属于原请求的后处理，计入 application/packet 延迟；不新增推理轮次。输出文本会增加相应响应字节，不能与旧版误标为 detection 的响应直接比较。
- `input_scale` 仍是传入合成 RGB 图片相对 224 像素基准的缩放倍率。模型内部可能缩放到固定分辨率；输出 token 数也不能代表视觉编码器 FLOP。此实现覆盖官方旧 pipeline 可加载的图像描述模型，不扩展到多模态对话或所有模型架构。
- 使用现有 CSV 列和任务相关的 `static_meta.json.output_format`，沿用现有输出字段；运行环境元数据见 static schema v7；窗口聚合沿用现有逻辑，只对有限的输出计数求平均，全部不可得时为 `nan`。缺少可选输出字段时为 `nan`，静态元数据必须为当前 schema v7。正式性能分析仍筛选 `status=ok` 且 `warmup=0`。
