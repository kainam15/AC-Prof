# 运行环境与配置

[← 返回专题目录](runtime.md)

## 当前配置

| 逻辑 profile | 选择条件 | 依赖与接口 |
| --- | --- | --- |
| `multimodal-transformers4576` | 已支持的原生 `multimodal` 模型 | Transformers 4.57.6；`family-default` handler |
| `<family>-transformers560-<platform>` | NLP、CV、Audio、Multimodal 中旧 Auto 注册表未覆盖、5.6.0 已登记的原生架构 | 按任务和 `model_type` 选择；四个任务族共享每个平台的完整环境，仍用 `family-default` |
| `moss-transformers560` | MOSS 官方模型 ID，或 `model_type=moss_transcribe_diarize` | Transformers 5.6.0；`moss-transcribe-diarize` adapter |
| `<family>-cu128` / `<family>-cu124` | NLP、CV、Audio、Diffusion、Structured、Timeseries | 完整依赖锁；Torch 2.11.0 / 2.6.0，保留对应任务族接口 |
| `<family>-cpu` | 显式 CPU 索引或容器 CI | CPU wheel；同样使用完整依赖锁 |
| `onnxruntime-cpu` | structured 的 ONNX dense tabular 接口 | ORT 1.23.2、NumPy；CPU float32 |
| `onnxruntime-cv-cpu` / `onnxruntime-nlp-cpu` | ONNX 图像分类／文本分类 | 共用同一个无 Torch、无 Transformers 环境；Pillow 图像处理、Tokenizers 本地分词 |

[`runtime_profiles.py`](../../acprof/runtime_profiles.py) 使用标准库声明三个独立对象：

| 对象 | 声明内容 |
| --- | --- |
| `RuntimeProfile` / `PROFILES` | 名称、任务族、adapter、模型/backend 约束、dtype、环境引用和版本线；当前共 40 个 profile。 |
| `PlatformSpec` / `PLATFORMS` | Linux amd64、固定 Python 基础镜像 digest、Python 3.10.21、系统锁；Torch 字段可省略。旧 CPU / cu128 为 Torch 2.11.0，cu124 为 2.6.0。 |
| `DependencyEnvironment` / `ENVIRONMENTS` | 平台引用及完整 Python 制品锁；当前有 27 个唯一环境；可用 `RuntimeSpec(type, version, package)` 核验运行时包的锁版本。名称仅用于引用，不决定内容身份。 |

`audio-cpu` 与 `multimodal-transformers4576-cpu` 共享 `audio-cpu` 环境；cu124 的对应两个
profile 共享 `audio-cu124` 环境。cu128 的 audio 使用 `tqdm==4.70.1`，原生 multimodal 使用
`4.70.0`，因此保留独立环境。MOSS 继续使用 Transformers 5.6.0 的专用 cu128 环境。
`transformers560-cpu/cu124/cu128` 是三个共享原生环境；复用 5.6.0 约束并加入 timm，不选择 MOSS adapter。
同族可有多个环境，不同族可共享环境；环境和 profile 数量均不要求长期保留同等数量的镜像。

完整锁位于 [`dockerfiles/locks`](../../dockerfiles/locks)，源约束位于
[`dockerfiles/requirements`](../../dockerfiles/requirements)。旧平台的 `platform-*.txt` 包含 Torch
必需依赖闭包和基础安装工具；`platform-python-cpu.txt` 只有基础安装工具。Flask、torchvision、torchaudio、NumPy、Pillow 等由环境完整锁声明。
每个包固定一个适用于目标 Python/ABI/架构的 wheel URL 和 SHA256，包括 pip、setuptools、wheel
及其依赖。各环境直接继承所声明的平台，不通过升级另一个环境来构建。CV 三个平台的锁新增
`timm==1.0.27`，因此环境身份改变；其余已有环境的包集保留。新运行时角色声明不改变相同完整
包集的身份；缺少包或锁版本不一致会失败。

系统锁 [`system-trixie-amd64.json`](../../dockerfiles/locks/system-trixie-amd64.json) 固定基础镜像、
Debian `20260912T203535Z` 和安全仓库 `20260912T113611Z` 的实际 snapshot URL、签名索引摘要、
全部直接/传递系统包版本和新增/升级 `.deb` 的 URL、大小、SHA256。基础镜像已有包由 OCI digest
固定，并计入最终完整包集合。普通构建只下载锁中的制品，校验哈希后通过 `--no-download` 安装，
不查询浮动 apt 仓库、不动态选择包名。

主机使用 [`requirements/host.lock`](../../requirements/host.lock)，支持 Python 3.10+。依赖集合与兼容区间只在
[`pyproject.toml`](../../pyproject.toml) 声明；[`requirements/host.in`](../../requirements/host.in) 是已验证版本约束，
安装直接指定 `requirements/host.lock`。主机、开发工具和测试依赖采用以下布局：

```text
AC-Prof/
├── pyproject.toml
├── requirements/
│   ├── host.in
│   ├── host.lock
│   ├── dev.in
│   ├── dev.lock
│   ├── test.in
│   ├── test.lock
│   ├── runtime-test.in
│   └── runtime-test.lock
├── acprof/
├── dockerfiles/
├── scripts/
└── tests/
```

以上只展示依赖相关目录。`dev.in` 和 `test.in` 通过相对路径 `-c host.lock`
约束共享包版本；安装和生成命令从仓库根目录执行。容器输入与制品锁仍位于 `dockerfiles/`。
锁更新工具固定为 uv 0.12.13。
生成目标 wheel 锁及检查主机 TOML 元数据需要 Python 3.11+；容器锁检查和运行代码仍支持 Python 3.10+。
主机检查离线核对 Python 3.10–3.14 的 marker 分支、直接依赖和 pin，不代表在这些解释器上运行过测试。

```bash
# 只读：锁格式、目标平台、源约束及 profile/环境映射；不访问 Docker 或网络
.venv/bin/python scripts/compile_locks.py --check
# 只读：主机声明、已验证约束与 lock 的版本一致性；需要 Python 3.11+
.venv/bin/python scripts/compile_locks.py --host-only --check
# 从 pyproject.toml 和已验证约束重新生成主机锁
.venv/bin/python scripts/compile_locks.py --host-only --uv .venv/bin/uv
# 保持当前全部包版本重新解析制品；--upgrade 才允许更新环境包
.venv/bin/python scripts/compile_locks.py --runtime-only --variant cpu --uv /path/to/uv
# 在固定基础容器中重新解析系统锁；只有此显式更新步骤运行 apt update
.venv/bin/python scripts/compile_system_lock.py --snapshot 20260913T000000Z
```

不带 `--check` 时，`--host-only` 只更新主机锁；`--variant` 可重复，省略时处理全部平台。迁移保留了原有全部 Python
包版本，仅补齐基础安装工具、目标制品及其哈希。更新锁后仍需执行目标容器验证。即使依赖和
来源完全锁定，也不宣称重建的 image ID 必然相同；复现实验和补采仍使用原始 image ID。

主机输入收敛复用 [uv 的 pyproject 与 constraints 编译方式](https://github.com/astral-sh/uv/blob/main/docs/pip/compile.md)。
沿用已有工具和锁格式，不引入新的包管理器或运行依赖；uv 提供 MIT 许可，解析仅在显式更新时运行。

主机构建预检沿用驱动兼容分支，CUDA 12.4 选择固定的 Torch 2.6.0 wheel，CUDA 12.8+ 选择 2.11.0。
`ACPROF_NLP_TORCH_INDEX_URL` 接受官方 `cu124`、`cu128` 和 `cpu` 索引；显式
`ACPROF_NLP_TORCH_SPEC` 必须与该分支的精确版本一致。其它组合需登记并验证自己的完整锁，
不再用无上界范围绕过锁。CPU 容器 CI 不证明 CUDA wheel 或所有模型的 GPU 兼容性。

支持任务标签不等于支持所有 checkpoint。已知不兼容的架构在任务预检退出；未登记的自定义
架构不会自动安装其 requirements 或执行主机端模型代码。通过静态检查的模型仍须完成实际
推理验证；CPU、GPU、dtype 和每种 profiler 的支持分别判断。
