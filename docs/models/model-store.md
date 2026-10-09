# 下载网络与 Model Store

[← 返回专题目录](runtime.md)

默认使用内部 `auto` 策略。先读取 Model Store 中相同 source、模型 ID 和 revision 的完整计划；命中后模型解析和准备不请求网络。未指定 revision 时按 HF 的 `main` 或 ModelScope 的 `master` 请求，只复用同一默认分支的已记录缓存；不能把此前指定的 tag 或 SHA 当作默认分支。显式 SHA 可匹配同一 commit，显式分支／tag 只匹配已记录的同名 ref。合适的完整缓存不联网检查分支最新提交。无完整缓存时先尝试 `HF_ENDPOINT`（默认 `https://hf-mirror.com`），正常跟随可信重定向；入口或存储传输失败后尝试备用入口及官方 `https://huggingface.co`，已经固定的 commit 在重试中保持不变。中国大陆用户通常只需选择模型、检查预计大小与预算、确认下载；“国内入口优先”不保证所有权重字节都来自国内服务器。

Hub endpoint 和 storage endpoint 分别处理。镜像可以重定向到官方 Hub、HF CDN 或 HTTP Xet bridge；可信存储按 Hugging Face 控制的 `hf.co`、`huggingface.co` 域名边界识别，允许新的区域子域，不枚举少数 CDN 机器名。不允许任意第三方、HTTP 降级、URL 内嵌凭据或异常端口，也不把通用 S3／CloudFront 域名整体加入信任范围。存储请求移除 Hub 的 Authorization／Cookie。原生 Xet 和 hf_transfer 通道禁用，以便通过公开 HTTP client factory 统一观察和检查；HTTP Xet bridge 仍可下载。未知域名明确报错，不永久标记模型不可用。

旧 CLI 的 `mirror-only`、`mirror-preferred`、`official` 仅保留为高级兼容参数：`mirror-only` 限制初始 Hub 入口，仍允许可信 CDN 重定向。TUI 的历史模式统一迁移为 `auto`。AC-Prof 不配置 VPN，不修改系统标准代理变量；历史项目配置中的代理字段读取后忽略。系统代理示例见 [CLI](../usage/configuration.md#主机环境与-hugging-face-认证)。

`network_policy.py` 的新报告使用 schema v2，记录 URL、host、category、cache 状态、估算和实际 bytes。应用层连接只区分 `direct-socket` 与 `explicit-proxy`；实际公网出口为 `unknown`／`externally-managed`，上游透明代理、VPN、NAT 由系统网络负责。OCI 的网络由 Docker 管理，不依据主机应用代理推断出口。不再产生 DIRECT／PROXY 流量分摊或预计 VPN 流量；历史报告保持原样。安装 AC-Prof 之前的 uv bootstrap 不属于 runtime 下载预算。

所有新构建先完成 Network Preflight：模型文件列表、总量、本地已有量、待下载量与 endpoint；Model Store 剩余空间和容量；平台/环境本地 image 命中、是否尝试 GHCR、OCI manifest 的压缩总量上界；精确 Python/Debian artifacts 的大小。HF 选一个待下载文件执行 HEAD probe，沿可信重定向检查存储可达性，不请求权重正文；完整缓存跳过该 probe。一次 CDN 失败只影响当前尝试，可重新预检／重试。BuildKit 缓存无法从主机可靠读取时显示 `unknown`，按完整制品大小保守估计；HEAD 或 manifest 不能给出大小时保留 `null`，不当作零。`expected_download_bytes` 在批量下载前显示，并随 runtime 元数据保存。

HF 的合理传输路径全部失败时显示失败阶段（Hub／redirect／storage）、DNS／TLS／HTTP／timeout 等原因及最终 host，提供重试、诊断和“改用 ModelScope”。仓库、revision、文件不存在或访问受限时保留对应模型错误，提示修正身份或权限，不统一包装成 VPN 问题。`HF_HUB_OFFLINE` 启用时仍在请求发送前阻止联网，已有完整缓存可继续使用。切换需要用户明确确认 ModelScope 模型 ID 和 revision，生成独立实验配置；不会自动复制同名模型或静默替换 source。`huggingface` 与 `modelscope` 是不同来源，同名、同 revision 文本不证明内容一致。ModelScope 使用官方轻量 `modelscope-hub` SDK；分支／tag 由 `git ls-remote` 固定为仓库 commit，文件按该 commit 下载并校验 SHA256，支持 SDK 的 partial／resume。主机需安装 Git，私有仓库还需自行配置相应 Git 认证；也可直接指定完整 commit SHA。ModelScope 能下载不等于 AC-Prof 支持其所有推理架构；已有显式 HF 依赖继续保留 HF 来源。

`--max-download 5GB`（十进制）或 `5GiB`（二进制）约束本次计划的批量 payload。总量未知或超限会在任何权重下载、OCI pull 或 dependency build 前退出；API、HEAD 和受限配置 JSON 查询是计划所需的小额流量，不是零流量预检。它不是 TCP/TLS 开销和失败重传的精确线速账单。预算模式下 GHCR 失败不继续未规划的本机构建；需显式选择 `ACPROF_RUNTIME_IMAGE_SOURCE=build` 后重新预检。GHCR 未被禁用，国内 OCI registry 可通过现有 `ACPROF_RUNTIME_REGISTRY` 显式选择，仍核验完整身份。

Model Store 默认位于 `~/.cache/acprof/model-store`，可用 `--model-store` 更改。`hf/` 与 `modelscope/` 按来源隔离权重缓存；`entries/<id>/hf/` 是供离线 loader 使用的相对链接视图，目录名不代表模型 source。各依赖保留独立 `refs/main`，避免不同 revision 互相覆盖。清单固定 source、model ID、完整 commit、文件大小、每文件 SHA256 与计划 SHA256。历史 HF entry 身份保持兼容；ModelScope entry 加入 source，不与同名 HF entry 混用。entry 身份不包含 CPU/GPU/profile，筛选器与 catalog 的变更会失效旧 entry。

entry 先在私有临时目录完整构建，再原子发布；发布前仅为 entry 顶层补足目录搜索（execute）权限，确保禁用 DAC override 的只读 Docker 验证容器也能访问快照，不开放目录列表。历史由临时目录以 `0700` 权限发布的 entry，在取得 Model Store 锁、验证固定清单后，于运行时挂载前修正顶层目录权限；不重下权重、不改模型文件权限，也不增加容器 Linux capabilities。

下载清单 schema v1 增加可选的 `source`、`requested_revision`、`repository_context` 和 `download_provenance`；历史缺 source 按既有 HF 语义读取，不重写已有实验。`repository_context` schema v1 保存原始 Hub 元数据和完整仓库文件列表，不能用下载筛选后的文件或用户覆盖的任务字段代替。历史缺少完整上下文的计划仍可按固定身份读取，不伪造仓库元信息；读取缓存配置文件先检查大小上限及 SHA256。Hub `endpoint` 是传输信息，不作为模型身份。`download_provenance` 记录入口、最终 endpoint 类型、失败尝试与去除签名 query／凭据的 redirect chain，最多保存 256 次响应并标记截断。完整缓存保留原记录，不虚构当次网络流量。SDK 无法暴露的实际 CDN 或历史缓存原始地址保留 `unknown`；目前 ModelScope SDK 不提供等价的 redirect hook，因此其存储链不推断为 Hub 地址。细节见[结果协议](../profiling/metadata.md#static_metajson-字段)。

主机不安装推理框架。`auto` 筛选使用固定 runtime lock 以及 `container/compat/transformers_model_types.json` 中 4.57.6/5.6.0 的 native model types，catalog 来自相应锁定 runtime 的公开 `CONFIG_MAPPING_NAMES`，保存源码位置与 SHA256；未知版本/布局保持完整快照。新增版本须重新提取并核验 catalog，不能借旧版本的支持表作推断。

模型校验、下载、磁盘检查和 LRU 更新在准备阶段完成。`plan_model` 命中缓存时保留只读快速路径；缓存未命中时，在重新检查缓存及下载、读取主模型和依赖仓库的规划元数据期间持有 Model Store 锁，防止并发 prune 删除尚无 entry 引用的文件；规划成功、失败或中断后释放锁，未引用文件仍可回收。容器只读挂载 Model Store；server、独立验证、profiler 和补采都传递固定 snapshot/cache 路径与离线环境变量，custom-code cache 放在可写 `/tmp`。缺失 store、计划不匹配或准备／补采前文件 SHA256 不匹配直接失败，不在线修复。主机结果另记 `model_store.host_path` 以便补采找到自定义目录，该路径不写入 Docker 镜像；迁移目录后可用 `ACPROF_MODEL_STORE` 显式覆盖。保持挂载期间的共享 lease，prune 不删除活动实验引用的 entries。测量样本、连接策略与窗口不变。

独立运行验证只有在 Docker 尚未发起，或按不可变容器 ID 确认删除/不存在后，才释放本次
Model Store 租约。已发起 Docker 但拿不到 ID、清理失败或清理被取消时保留保护，不能把异常
当作容器已消失。同一次验证中，较晚取得的合法 ID 若已完成清理，可解除先前的清理阻塞；
原始验证超时仍记录为 `inconclusive`，不会改成运行成功。

停止正式运行容器失败或被取消时，其租约由当前进程继续强引用，即使上层丢弃会话对象也不会
提前解除 GC 保护。重复失败只保留一次；按不可变 ID 确认清理成功后关闭对应租约并移除保留记录，
其他消费者的独立租约不受影响。描述符关闭失败保留重试句柄，不把未完成关闭标为已完成。
该保留表仅在当前进程内有效，不能替代进程崩溃／退出后的持久化消费者身份和清理恢复。

```bash
acprof run --model google-bert/bert-base-uncased --max-download 5GB
acprof run --model-source modelscope --model Qwen/Qwen3-0.6B --max-download 5GB
acprof model-store status
acprof model-store prune --target-size 100GB              # 预览 LRU 删除及可回收量
acprof model-store prune --target-size 100GB --apply      # 显式执行
```

`--model-store-max` 在下载前核对总容量；同时检查文件系统 free space 并预留 64 MiB 元数据余量。大小未知、容量不足或磁盘不足直接停止。清理为显式操作，按最久未使用顺序选择，保留活动 lease 和 `--keep <entry-id>`。可回收量按实际共享 blobs 计算；多个模型的逻辑大小不能直接相加当成物理磁盘用量。旧 baked 镜像可能仍被历史实验/补采引用，应通过镜像管理明确清理；新构建不再向 Docker 写入第二份权重。

目标容量预览只扫描一次 entry 与 blob 引用关系，在内存中按 LRU 顺序减少引用计数；
最后一个引用被选入删除范围后才累计该 blob 的可回收字节。容量统计与引用扫描次数固定，
不会随淘汰条目数重复遍历候选集合。孤立 blobs 仍计入可回收量，活动 lease 与共享引用继续保留。

预览不授权直接执行旧删除计划。实际清理重新取得全局 store lock，在锁内核验当前 entries、
lease 和共享 blob 引用；TUI 同时限定在用户确认的 entry 集合内，预览后新增的模型不进入删除集。
等待锁和扫描可以协作取消，尚未开始删除时不改动权重；已开始删除则持锁完成收尾后释放界面任务状态。
这些工作只在准备或显式存储管理期间执行，正式测量窗口不扫描缓存。

缓存候选、容量摘要和准备阶段共用 `read_entry` 读取已验证计划：只接受常规元数据文件，
使用 `ENTRY_METADATA_MAX_BYTES` 限制为 4 MiB，打开后核对大小并再次有界读取。
`entries`、entry 目录和计划文件不能使用 symlink；用户配置的 Store 根路径仍可为 symlink。
目录逐级按已打开的父目录解析，防止并发替换将读取导向 Store 外；缺失 entry 返回空，损坏或超限明确报错。
该读取不获取清理锁、不扫描 blobs，也不重新计算权重 hash。

此设计参考 Hugging Face Hub 的 [缓存扫描与删除策略](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_cache_manager.py)
（Apache-2.0，持续维护），借用先扫描、再生成共享文件删除计划的思路。AC-Prof 保留自己的
固定计划视图和活动 lease，使用现有标准库实现引用计数与可取消锁等待，不新增缓存框架或依赖。

Python/CUDA runtime build 消费精确 wheel URL，`PYPI_MIRROR_INDEX` 不能改写这些 URL。`scripts/compile_locks.py --index-url <https-index> --runtime-only --variant cpu` 让固定 uv 重新从指定索引解析目标平台 artifacts；`--torch-index-url` 可显式指定相应 CUDA wheel 索引。默认保留 exact versions；同版本 artifact SHA256 改变会拒绝替换。生成 `.artifacts.json` 保存目标平台、来源与已知大小。不要字符串替换 URL。`--check --variant cpu --variant cu124 --variant cu128` 是离线锁校验，不能代替三个环境的实际构建/推理。

保留 `/root/.cache/pip` 和 `/root/.cache/acprof/debs` 的 BuildKit cache mounts。wheel 使用 pip cache 目录下按 SHA256 寻址的 `acprof-artifacts/`，校验命中后完全跳过网络；未命中时检查重定向、验证 hash，再以本地 URL 交给 `pip --no-index --no-deps --require-hashes`。旧 pip HTTP cache 不自动转换为新缓存，首次迁移应按完整依赖下载量预检；已有 `.deb` hash 缓存可直接复用。依赖缓存键不含业务源码或权重；普通业务修改复用依赖层。构建日志报告平台/环境与每个 artifact 的 hit/miss、节省量、新下载 payload 和实际来源 host，写入 `build_download_sources`；该字段是镜像构建 provenance，不是本次 OCI pull 流量。Docker 未暴露精确传输字节时保留未知。模型日志中的 `verified_new_payload_bytes` 是新增完整文件的逻辑大小，`wire_bytes` 仍为未知。

依赖下载及大小预检统一使用 `acprof-dependency-downloader/1.0` 的 `User-Agent`，避免官方制品源拒绝默认 `Python-urllib` 客户端。预检在重定向后仍保持 HEAD；HTTP 错误返回未知大小，预算检查继续拒绝未知总量。客户端标识不会改写锁定 URL、SHA256 或放宽来源切换策略。做法参考 [PyTorch 的 `torch.hub` 下载器](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/hub.py)（BSD 风格许可证），沿用现有标准库 `urllib`，没有新增运行依赖或测量窗口开销。

构建时的 wheel / `.deb` GET 遇到 TLS EOF、连接中断、超时或 `IncompleteRead` 时，
最多尝试 3 次，重试前分别等待 1、2 秒；每次保持同一锁定 URL、客户端标识和重定向策略，
丢弃未完成的 `.part` 后重新下载，完整 SHA256 核验通过才发布缓存。耗尽次数后传播原错误；
HTTP、证书、来源策略、磁盘权限和哈希错误不重试。成功记录中的 payload bytes 是最终通过核验的文件大小，
不包含失败尝试的未知传输量，`wire_bytes` 仍为未知。
有限重试参考 [pip 网络会话](https://github.com/pypa/pip/blob/main/src/pip/_internal/network/session.py)
与 [urllib3 Retry](https://github.com/urllib3/urllib3/blob/main/src/urllib3/util/retry.py) 的故障分类和退避思路
（MIT），并核对 [pip Issue #11843](https://github.com/pypa/pip/issues/11843) 与
[PR #12500](https://github.com/pypa/pip/pull/12500) 对瞬时错误的取舍；这里仅重试传输错误，
沿用标准库实现，不引入 pip 私有 API、额外依赖或测量窗口内重试。

旧 `python -m acprof.container.download_model` 下载入口已停用并明确报错，避免绕过主机预检、容量检查和预算。

实现参考 [Hub client factory](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_http.py)、[Hub HEAD 重定向 PR #4739](https://github.com/huggingface/huggingface_hub/pull/4739)、[官方 ModelScope Hub SDK](https://github.com/modelscope/modelscope_hub)、[uv 索引规则](https://github.com/astral-sh/uv/blob/main/docs/concepts/indexes.md) 和 [BuildKit cache mounts](https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md)。借用公开 transport、Hub／storage 分层、SDK 断点续传和固定哈希校验；Hub/ModelScope/BuildKit 为 Apache-2.0，uv 为 MIT/Apache-2.0。ModelScope 仅新增主机侧轻量 SDK，不引入推理框架；公开 factory 的适配由真实 SDK mock transport 回归保护。所有诊断和下载均在测量窗口外。
