# 运行时安全基线

[← CI 与审计入口](ci.md)

本记录针对 2026-10-24 的 Python 3.12.15 依赖基线。锁、制品身份、兼容性验证和漏洞状态
分别记录；构建成功、推理通过或扫描器跳过某个包，都不能证明该包安全。以下剩余问题是
待处理风险，不表示维护者已经接受风险，也不替代正式发布审批。

## Python 包审计

4 个平台锁和 27 个环境锁共包含 153 个不同的包／版本组合。pip-audit 的分类为：

| 范围 | 无已知漏洞 | 发现漏洞 | 未审计 |
| --- | ---: | ---: | ---: |
| 完整锁 | 2 | 26 | 3 |
| 包／版本 | 142 | 2 | 9 |

锁的分类优先显示已知漏洞；有漏洞的锁也可能包含未审计 wheel。仅当全部包均已审计且
没有发现漏洞时，才能标为“无已知漏洞”。两个完全无已知漏洞的锁为 `platform-python-cpu`
和 `onnxruntime-cpu`。`platform-cpu`、`structured-cpu` 和 `transformers5104-cpu` 仍有未审计 wheel。
跨锁和 advisory alias 按连通关系去重后，Python 包扫描剩余 **8 个不同漏洞**；
不得把各锁的漏洞数相加，也不得将 PYSEC、GHSA、CVE 别名重复计数。

当前固定 Pillow 13.0.0、filelock 4.2.4、pip 26.2.2、Flask 3.1.3、ONNX 1.22.0、
sentence-transformers 5.6.0；原 Transformers 5.6.0 环境升级为 5.10.4。
CPU / 无 Torch 平台采用 setuptools 84.1.0。上述版本在本次 pip-audit 中无已知漏洞，
不代表未公开问题不存在。特别是 [sentence-transformers 5.6.0](https://github.com/huggingface/sentence-transformers/releases/tag/v5.6.0)
对本地自定义代码的信任检查仍有 warning 路径，不能将其描述为强制安全沙箱。

| 剩余依赖 | 去重标识 | 处理和范围 |
| --- | --- | --- |
| Transformers 4.57.6 | CVE-2025-14929 | X-CLIP checkpoint 转换反序列化；常规推理不执行转换脚本，仍保留扫描发现。 |
| Transformers 4.57.6 | CVE-2026-1839 | Trainer RNG 恢复路径；本项目不训练，现有 Torch 均不低于 2.6，但不能据此删除整个依赖的漏洞记录。 |
| Transformers 4.57.6 | CVE-2026-4372、CVE-2026-5241 | 模型配置和嵌套加载可绕过代码信任检查；属于需要继续处理的加载风险，离线设置不等于恶意配置审计。 |
| Transformers 4.57.6 | CVE-2026-9856、CVE-2026-80047 | `save_pretrained` 路径穿越、custom generation 代码下载顺序；不向未受信任模型开放这些能力，保留风险记录。 |
| GPU setuptools 81.0.0 | CVE-2026-59890 / GHSA-h35f-9h28-mq5c、GHSA-xmj3-9gf7-h52w | 公告涉及 macOS sdist 名称和 Windows legacy egg 安装路径。Linux 推理镜像仅安装校验哈希的 wheel；cu128 的 Torch 2.11 要求 `<82`，不能强行改锁到 84.1。cu124 的 Torch 2.6 没有这项上限，但升级兼容验证随该平台正式验收暂缓，保留未修复记录。 |

4.57.6 是本次检查可用的最后一个 4.x 版本；5.x 已移除 image-to-image、image-to-text、
question-answering、summarization、text2text-generation、translation、visual-question-answering
等 pipeline。迁移这些任务的接口后才能删除 4.x 环境，不能用升级导致的任务丢失换取扫描通过。
5.10.4 的原生软链接加载尚未通过，继续使用[已验证的兼容路径](../models/pipelines.md)。

setuptools 范围依据[macOS 公告](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c)、
[Windows 公告](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w)和
[Torch 2.11 依赖声明](https://github.com/pytorch/pytorch/blob/v2.11.0/pyproject.toml)。
扫描数据库与公告的修复版本存在差异时，两者均保留，不据此改写为“已修复”。

## 特殊 PyTorch wheel

9 个 CPU/CUDA `torch`、`torchvision`、`torchaudio` 包／版本组合无法由本次 PyPI 扫描完整评估。
独立核对官方索引中的精确文件名和 SHA256；镜像构建仍按清单校验实际下载字节。
这证明制品身份，不构成全量安全审计。

- CPU 升级到 Torch 2.14.1 / torchvision 0.29.1；TorchAudio 保持 2.11，遵循
  [上游稳定 ABI 的兼容约定](https://github.com/pytorch/audio/blob/main/docs/source/installation.rst)。
- cu128 保持 Torch 2.11。它不在 [CVE-2026-24747](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
  的受影响版本范围内，但仍被 CVE-2025-3000 的 JIT 条目命中；不要描述为“无已知漏洞”。
  [上游问题与修复讨论](https://github.com/pytorch/pytorch/issues/149623)及扫描数据库列出的修复版本需持续跟踪。
  [PyTorch 2.12](https://pytorch.org/blog/pytorch-2-12-release-blog/) 起的 CUDA 发布范围不能直接替代 cu128。
- cu124 的 Torch 2.6 仍受 CVE-2026-24747、libuv TensorPipe 公告等问题影响，当前正式验收暂缓。
  保留锁用于明确旧环境身份，不计入已验收平台。

## OCI 与系统层

Python 基础镜像固定到 3.12.15 的 Linux amd64 OCI manifest，Debian 和 security snapshot
分别固定为 `20261023T202718Z`、`20261023T214343Z`。扫描使用独立 Grype 0.121.0 和
2026-10-23 的数据库，针对实际 image ID；保留原始 JSON 的匹配方式、包位置、修复状态和数据库时间。
本次扫描 14 个平台／环境镜像，跨镜像和 advisory alias 合并后共 196 个漏洞组；
这与上面的 Python 包审计范围不同，不能相加。Debian 包的匹配均为 `wont-fix` 或
`not-fixed`；数据库列出修复版本的条目来自 CPython、Python 依赖和内嵌 FFmpeg。
这些结果均未作为已接受风险豁免。

OpenCV 4.13 wheel 内嵌 FFmpeg 8.0.1，更新 Debian 包不能修复该副本。
升级为 [OpenCV 4.14.0.94](https://github.com/opencv/opencv-python/releases/tag/94)，其中 FFmpeg 8.1.2
修复 CVE-2026-8461 等问题。需要更新 FFmpeg 8.1.3 / 9.x 的后续条目仍须保留；
没有采用未锁定二进制或未经兼容验证的 OpenCV 5.x 替换它。

Python 3.12.15 的扫描条目不能因为修复列表只出现 3.13 / 3.14 / 3.15 就自动排除：

- CVE-2025-12781 / CVE-2026-3446 涉及 Base64 解码；外部请求和模型声明入口使用
  `validate=True`，没有使用 URL-safe / alternate alphabet。内部生成素材的解码不构成任意输入安全保证。
- CVE-2025-15366 / CVE-2025-15367 涉及 IMAP / POP3，CVE-2026-6019 涉及 cookie JavaScript 输出；
  当前采集入口未使用这些路径，保留包层发现。
- CVE-2026-12345 涉及临时目录清理竞态；只使用当前用户的私有临时目录及非 root 推理容器，
  仍需跟踪 [CPython 修复](https://github.com/python/cpython/commit/06ef3d43fdaf4da97af12fa694b4886092c4b970)
  的 3.12 可用性，不能称为已修复。
- CVE-2026-3479 的 `pkgutil.get_data` 争议条目与其他无可用发行版修复的系统包分别记录；
  不修改扫描器 ignore 规则来获得零告警。

OCI 中同一个漏洞可能同时匹配多个 Debian 二进制包和内嵌库；原始匹配数、去重漏洞数、
包／版本数分别统计。`fixed` 仅表示扫描数据库给出了修复版本，须进一步确认目标发行版或
wheel 是否提供该版本；`wont-fix`、`not-fixed`、`unknown` 都不等于已接受风险。
