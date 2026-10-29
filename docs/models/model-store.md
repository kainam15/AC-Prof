# 下载网络与 Model Store

[← 返回专题目录](runtime.md)

默认使用内部 `auto` 策略。先读取 Model Store 中相同 source、模型 ID 和 revision 的完整计划；命中后模型解析和准备不请求网络。未指定 revision 时按 HF 的 `main` 或 ModelScope 的 `master` 请求，只复用同一默认分支的已记录缓存；不能把此前指定的 tag 或 SHA 当作默认分支。显式 SHA 可匹配同一 commit，显式分支／tag 只匹配已记录的同名 ref。合适的完整缓存不联网检查分支最新提交。无完整缓存时先尝试 `HF_ENDPOINT`（默认 `https://hf-mirror.com`），正常跟随可信重定向；入口或存储传输失败后尝试备用入口及官方 `https://huggingface.co`，已经固定的 commit 在重试中保持不变。中国大陆用户通常只需选择模型、检查预计大小与预算、确认下载；“国内入口优先”不保证所有权重字节都来自国内服务器。

Hub endpoint 和 storage endpoint 分别处理。镜像可以重定向到官方 Hub、HF CDN 或 HTTP Xet bridge；可信存储按 Hugging Face 控制的 `hf.co`、`huggingface.co` 域名边界识别，允许新的区域子域，不枚举少数 CDN 机器名。不允许任意第三方、HTTP 降级、URL 内嵌凭据或异常端口，也不把通用 S3／CloudFront 域名整体加入信任范围。存储请求移除 Hub 的 Authorization／Cookie。原生 Xet 和 hf_transfer 通道禁用，以便通过公开 HTTP client factory 统一观察和检查；HTTP Xet bridge 仍可下载。未知域名明确报错，不永久标记模型不可用。

旧 CLI 的 `mirror-only`、`mirror-preferred`、`official` 仅保留为高级兼容参数：`mirror-only` 限制初始 Hub 入口，仍允许可信 CDN 重定向。TUI 的历史模式统一迁移为 `auto`。AC-Prof 不配置 VPN，不修改系统标准代理变量；历史项目配置中的代理字段读取后忽略。系统代理示例见 [CLI](../usage/configuration.md#主机环境与-hugging-face-认证)。

`network_policy.py` 的新报告使用 schema v2，记录 URL、host、category、cache 状态、估算和实际 bytes。应用层连接只区分 `direct-socket` 与 `explicit-proxy`；实际公网出口为 `unknown`／`externally-managed`，上游透明代理、VPN、NAT 由系统网络负责。OCI 的网络由 Docker 管理，不依据主机应用代理推断出口。不再产生 DIRECT／PROXY 流量分摊或预计 VPN 流量；历史报告保持原样。安装 AC-Prof 之前的 uv bootstrap 不属于 runtime 下载预算。

Docker 新构建和 Process 准备在安装依赖及下载完整模型前完成 Network Preflight：模型文件列表、缓存、待下载量、endpoint、Model Store 空间，以及 Runtime 下载来源。Docker 查询缺失平台/环境的 OCI manifest，记录压缩大小上界；本地环境命中时跳过无须访问的来源。Process 按实际 uv lock URL 规划 Python 依赖，不套用 BuildKit 镜像设置；完整环境通过既有身份和内容校验后跳过依赖网络检查。uv interpreter 未就绪且无法确定其制品时保留来源/大小未验证。

HF 从每个缺文件的仓库选择一个代表性大文件，先 HEAD，再对固定 revision 执行流式 Range GET；所有仓库、备用入口和失败尝试共享 **256 KiB 响应体读取上限**。沿用 SDK client、认证、系统代理及可信重定向检查；逐跳关闭响应，不读取重定向正文。服务器忽略 Range 时仍提前关闭流，不下载整个文件。每次 raw read 前为 HTTPX HTTP/1.1 的 64 KiB 缓冲保留余量，因此实际探测量可能小于上限；剩余预算不足一次 read 时跳过 GET。读取设置单次 timeout 和总耗时检查，输出 HTTP 状态、目标 host、接收 bytes、耗时和 DNS/TLS/连接超时/读取超时/HTTP/中断/重定向原因。完整模型缓存零请求；探测数据不落盘，不创建缓存条目或性能结果。ModelScope 的限量传输探测暂记“未验证”，不把未测试来源标为通过。

Python/Debian 对每个待用来源执行代表性 HEAD；Docker manifest 通过现有 Docker 命令及认证配置读取。二者只证明元数据可达，**不证明 Docker/BuildKit 内部网络或完整制品传输成功**。BuildKit/uv 制品缓存不可可靠确认时仍为 `unknown`，按完整大小保守估计；大小未知保留 `null`。失败证据直接进入下载确认页，用户可终止或明确确认后尝试正式下载，不能绕过预算。`expected_download_bytes` 表示完整下载计划，`probe_payload_bytes` 单独记录已读取探测正文，并一起参与预算核对；报告保存在 Runtime 元数据，测量窗口不执行探测。

HF 的合理传输路径全部失败时显示失败阶段（Hub／redirect／storage）、DNS／TLS／HTTP／timeout 等原因及最终 host，提供重试、诊断和“改用 ModelScope”。仓库、revision、文件不存在或访问受限时保留对应模型错误，提示修正身份或权限，不统一包装成 VPN 问题。`HF_HUB_OFFLINE` 启用时仍在请求发送前阻止联网，已有完整缓存可继续使用。切换需要用户明确确认 ModelScope 模型 ID 和 revision，生成独立实验配置；不会自动复制同名模型或静默替换 source。`huggingface` 与 `modelscope` 是不同来源，同名、同 revision 文本不证明内容一致。ModelScope 使用官方轻量 `modelscope-hub` SDK；分支／tag 由 `git ls-remote` 固定为仓库 commit，文件按该 commit 下载并校验 SHA256，支持 SDK 的 partial／resume。主机需安装 Git，私有仓库还需自行配置相应 Git 认证；也可直接指定完整 commit SHA。ModelScope 能下载不等于 AC-Prof 支持其所有推理架构；已有显式 HF 依赖继续保留 HF 来源。

`--max-download 5GB`（十进制）或 `5GiB`（二进制）约束本次计划的批量 payload 及限量模型 GET。先为完整下载预留预算，再使用余量探测；总预算或模型预算为零时不发送模型 GET。总量未知或超限在 payload 传输、OCI pull 或 dependency build 前退出；API、HEAD 和受限配置 JSON 查询仍产生元数据流量。响应体上限不是 TCP/TLS、内核预读及失败重传的精确线速账单。预算模式下 GHCR 失败不继续未规划的本机构建；需显式选择 `ACPROF_RUNTIME_IMAGE_SOURCE=build` 后重新预检。GHCR 未被禁用，国内 OCI registry 可通过现有 `ACPROF_RUNTIME_REGISTRY` 显式选择，仍核验完整身份。

探测设计参考 [aria2 dry-run 的 HEAD 检查](https://github.com/aria2/aria2/blob/master/src/CreateRequestCommand.cc)、
[HTTPX 流式响应及显式关闭](https://github.com/encode/httpx/blob/master/httpx/_models.py)、
[HTTPcore HTTP/1.1 有界读取](https://github.com/encode/httpcore/blob/master/httpcore/_sync/http11.py)
与 [HTTPX PR #2148](https://github.com/encode/httpx/pull/2148) 的流取消清理问题。
aria2 为 GPL-2.0-or-later，HTTPX/HTTPcore 为 BSD-3-Clause；仅借鉴检查、限量读取和关闭资源的思想，
不复制下载器代码、不新增依赖，使用现有锁定版本并通过本地 HTTP 服务验证边界。

Model Store 默认位于 `~/.cache/acprof/model-store`，可用 `--model-store` 更改。`hf/` 与 `modelscope/` 按来源隔离权重缓存；`entries/<id>/hf/` 是供离线 loader 使用的相对链接视图，目录名不代表模型 source。各依赖保留独立 `refs/main`，避免不同 revision 互相覆盖。清单固定 source、model ID、完整 commit、文件大小、每文件 SHA256 与计划 SHA256。历史 HF entry 身份保持兼容；ModelScope entry 加入 source，不与同名 HF entry 混用。entry 身份不包含 CPU/GPU/profile，筛选器与 catalog 的变更会失效旧 entry。

entry 先在私有临时目录完整构建，再原子发布；发布前仅为 entry 顶层补足目录搜索（execute）权限，确保禁用 DAC override 的只读 Docker 验证容器也能访问快照，不开放目录列表。历史由临时目录以 `0700` 权限发布的 entry，在取得 Model Store 锁、验证固定清单后，于运行时挂载前修正顶层目录权限；不重下权重、不改模型文件权限，也不增加容器 Linux capabilities。

容量查询只将 64 位十六进制 ID 目录中的已验证清单列为模型；`.preparing-*` 暂存内容仍计入占用，
但不作为已发布模型读取。暂存清单尚未写完或进程退出留下完整暂存视图，都不会阻断容量查询和按目标清理；
正式 entry 的损坏清单仍明确报错。

显式重新准备遇到缺失清单时，会在 store lock 内取得该 entry 的排他 lease，重新校验权重并构建完整视图；
活动消费者尚未释放 lease 时，在下载前报错。已有目录不能代替有效清单，也不会直接返回准备成功。
替换时先将旧视图移至同层暂存目录，保持相对权重链接有效；新视图发布前失败会恢复旧目录。
发布后的目录同步失败会报错并保留旧视图；进程突然退出留下的暂存视图也交由后续显式清理回收。
新准备可以复用通过校验的缓存权重，运行、resume 和补采的缺失清单检查仍直接失败，不在线修复。
此流程参考 [uv 的暂存目录发布与独立制品存储](https://github.com/astral-sh/uv/blob/main/crates/uv-cache/src/lib.rs)
（MIT，持续维护）；沿用 AC-Prof 的目录身份、lease 和标准库实现，不引入新的缓存布局或依赖。

有效清单不等于完整的 loader 视图。准备和实验前验证还检查主模型及每个依赖的 `refs/main`、
固定 revision 快照、文件大小，以及指向 Store 内共享缓存的直接相对文件链接。
refs 只按预期 commit 长度有界读取；目录符号链接和读取权限错误明确失败。
显式准备可从已经校验的源缓存重建缺失链接、错误 refs 或不符合布局的文件，仍需排他 lease；
已有清单有效时保留原计划 SHA256 和 provenance，不伪造新的下载记录。
完整视图的 SHA256 直接针对运行时使用的 entry 文件校验，单次准备不重复 hash 源缓存和 entry 两份路径。
内容不匹配仍失败，不静默改写或在线修复；实验前验证只报告不完整视图，不执行重建。
这些检查位于准备和已有的实验前验证阶段，运行时取得挂载 lease 的启动路径不增加目录扫描。
完整性核对参考 [HF 离线快照检查](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_snapshot_download.py)
（Apache-2.0，持续维护）；AC-Prof 使用固定清单检查自身视图，保留逐文件 SHA256 校验，不新增依赖。

完整 entry 也可独立于 SDK 源快照索引复用。预检、准备和本地候选扫描传入已知 entry ID，
仅在磁盘清单与待用计划完全一致、主模型及全部依赖的视图完整时，将这些文件计为缓存命中；
仅有相同摘要字符串不能替代清单内容核对。此时不探测模型网络，也不调用 SDK 重新查找源快照。
空间与下载预算使用同一次缓存观察，准备在 Model Store 锁内重新检查，预检后文件丢失仍受原下载预算约束。
完整视图随后逐仓库校验一次实际 payload SHA256，内容损坏明确失败；缓存命中不等于推理验证通过。
候选扫描使用清单所在的原 entry ID，支持历史 entry，仍保持只读且不 hash 权重。
缺失或不完整视图沿用源缓存检查与显式修复流程，不因为 SDK 索引缺失自动增加在线下载。
完整性判断参考上述 HF 源码及 [Issue #4624](https://github.com/huggingface/huggingface_hub/issues/4624)；
AC-Prof 使用自身已验证文件清单，不要求历史缓存具有新版 SDK 的 tree listing，不新增依赖或测量窗口内工作。

下载清单 schema v1 增加可选的 `source`、`requested_revision`、`repository_context` 和 `download_provenance`；历史缺 source 按既有 HF 语义读取，不重写已有实验。`repository_context` schema v1 保存原始 Hub 元数据和完整仓库文件列表，不能用下载筛选后的文件或用户覆盖的任务字段代替。历史缺少完整上下文的计划仍可按固定身份读取，不伪造仓库元信息；读取缓存配置文件先检查大小上限及 SHA256。Hub `endpoint` 是传输信息，不作为模型身份。`download_provenance` 记录入口、最终 endpoint 类型、失败尝试与去除签名 query／凭据的 redirect chain，最多保存 256 次响应并标记截断。完整缓存保留原记录，不虚构当次网络流量。SDK 无法暴露的实际 CDN 或历史缓存原始地址保留 `unknown`；目前 ModelScope SDK 不提供等价的 redirect hook，因此其存储链不推断为 Hub 地址。细节见[结果协议](../profiling/metadata.md#static_metajson-字段)。

摘要匹配后仍须校验清单字段：主模型与依赖共用 repository ID 和固定 commit 约束，
禁止会与目录编码冲突的 `--`；文件必须是规范的相对路径，不能是 `.` 或包含 NUL。
准备入口在配置传输和创建缓存目录前验证计划，文件校验入口在读取 payload 前验证计划；
从磁盘读取的计划同样拒绝非法身份、路径及不一致的字节汇总，不自动修正或重新计算摘要后接受。
合法历史 v1 清单保持原值与可选字段缺省语义。这些检查使用标准库和已有清单，不新增依赖或权重扫描。
设计参考 [HF 的 repository ID 校验](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_validators.py)
及其 [ASCII 约束修复 PR #4892](https://github.com/huggingface/huggingface_hub/pull/4892)
（Apache-2.0，持续维护）；复用边界校验思想，AC-Prof 的共享规则负责本地身份与路径安全，服务端命名规则仍由对应 SDK 核对。

逐文件校验先暂存实际大小与 hash，整个仓库校验成功后才更新计划及摘要；后续文件缺失、
内容错误、读取失败或取消不会留下部分更新的计划。模型准备使用独立的计划副本，主模型成功后，
依赖下载、校验或发布失败也不改变调用方的输入；原始异常继续向上传递，修复失败原因后可用原计划重试。
成功后的完整清单与 provenance 从返回回执读取，调用方不能依赖输入计划被修改。
已下载文件沿用既有缓存与清理规则，不因准备失败而删除；历史清单 schema、hash 算法及已发布结果保持原样。
设计参考 [HF 下载完成后发布临时文件](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py)
（Apache-2.0，持续维护）的暂存与提交边界；只借鉴思想，不复制源码或新增依赖。
新增开销限于准备阶段的元数据副本，不复制模型权重，也不增加 payload 扫描或测量窗口内工作。

有效清单还必须属于本次任务。缓存命中、锁内再次读取、准备和镜像复用共用请求身份核对：
source、model ID、固定 commit、下载策略、任务族、backend、adapter，以及包含 `allow_patterns` 的依赖声明。
身份冲突明确失败并保留原清单，不自动回退下载或覆盖；直接准备在传输配置和缓存写入前检查传入计划。
复用已有条目时，先选定已验证清单，再计算该清单的空间、下载预算和回执，避免检查与执行使用不同计划。
历史缺 source 仍按 HF 读取；endpoint、用户输入的分支名及执行设备不替代已固定的模型身份。
核对只比较现有元数据，不重复计算计划或权重 hash。设计参考
[pip 缓存对制品名称与请求名称的核对](https://github.com/pypa/pip/blob/main/src/pip/_internal/cache.py)
（MIT，持续维护），使用项目现有字段与标准库实现，不新增依赖或测量窗口内的工作。

主机不安装推理框架。`auto` 筛选使用固定 runtime lock 以及 `container/compat/transformers_model_types.json` 中 4.57.6/5.10.4 的 native model types，catalog 来自相应锁定 runtime 的公开 `CONFIG_MAPPING_NAMES`，保存源码位置与 SHA256；未知版本/布局保持完整快照。新增版本须重新提取并核验 catalog，不能借旧版本的支持表作推断。

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

`--model-store-max` 在下载前核对总容量；同时检查文件系统 free space 并预留 64 MiB 元数据余量。大小未知、容量不足或磁盘不足直接停止。清理为显式操作，按最久未使用顺序选择，保留活动 lease 和 `--keep <entry-id>`。可回收量包括将删除的 entry 常规文件与共享 blobs；多个模型的逻辑大小不能直接相加。容量与可回收量均按常规文件长度统计，不重复累计符号链接指向的内容，也不等同于文件系统实际释放的存储块。旧 baked 镜像可能仍被历史实验/补采引用，应通过镜像管理明确清理；新构建不再向 Docker 写入第二份权重。

目标容量预览只扫描一次 entry 文件与 blob 引用关系，在内存中按 LRU 顺序减少引用计数；
选中 entry 时累计其清单、refs 等常规文件，最后一个引用被选入删除范围后才累计该 blob 的可回收字节。
这样达到容量目标后不会因漏算元数据继续淘汰模型。容量统计与引用扫描次数固定，
不会随淘汰条目数重复遍历候选集合。孤立 blobs 仍计入可回收量，活动 lease 与共享引用继续保留。
entry 清单缺失时，仍从现存文件链接统计引用，并核对 lease、`--keep` 和批准集合；
清单缺失不能作为权重已无人使用的证据，也不会解除已有消费者的保护。
缺失清单且未受保护的目录可以在显式清理中整体回收，包括中断发布遗留的 `.preparing-*` 目录；
其残留常规文件计入回收量，共享权重仍需等最后一个保留引用消失才可删除。
清单缺失不会放宽新消费者挂载时的计划完整性校验。

预览不授权直接执行旧删除计划。实际清理重新取得全局 store lock，在锁内核验当前 entries、
lease 和共享 blob 引用；TUI 同时限定在用户确认的 entry 集合内，预览后新增的模型不进入删除集。
TUI 执行时沿用预览的容量目标，按锁内的当前占用重新决定还需淘汰哪些批准条目；
其他任务已经释放空间时，保留不再需要淘汰的模型。未设置目标仍清理所有批准且未使用的条目，
显式 `0` 仍表示零容量目标；未引用文件继续按既有规则回收。
清理时固定已打开的 Store 根目录，逐级检查 `entries`、HF 和 ModelScope 缓存的目录组件，
拒绝其中的目录符号链接；配置的 Store 根路径本身仍允许是符号链接，普通权重文件链接继续支持。
实际删除相对于已打开的父目录执行，防止检查后替换路径把删除重定向到 Store 外。
在任何删除开始前复核整个删除计划的路径；扫描、取消和错误路径均关闭目录描述符。
若其他程序在删除期间修改缓存而导致操作失败，会明确返回错误；已经完成的删除不回滚。
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
缺失清单的处理也参考 [Hub PR #4797](https://github.com/huggingface/huggingface_hub/pull/4797)
对引用证据不足时保留 payload 的原则；AC-Prof 直接扫描 entry 链接，并结合自己的 lease 决定保护范围，
不依赖下载清单推断权重无人使用。
目录边界借鉴 [CPython `shutil.rmtree` 的目录描述符实现](https://github.com/python/cpython/blob/3.12/Lib/shutil.py)
及 [Issue #46010](https://github.com/python/cpython/issues/46010) 的符号链接删除问题（PSF License）；
调用标准库公开的 `dir_fd` API，不复制删除器，也不增加正式测量窗口内的扫描。

Python/CUDA runtime build 默认消费锁定的精确 wheel URL，`PYPI_MIRROR_INDEX` 不能改写已锁定的 URL。国内受限流量场景可显式设置 `ACPROF_PYPI_ARTIFACT_MIRROR=tencent`：仅在**下载传输层**将 `https://files.pythonhosted.org/packages/...` 映射为 `https://mirrors.cloud.tencent.com/pypi/packages/...`，锁文件中的官方 URL、版本及 SHA256 保持不变；整文件 SHA256 通过后才发布缓存。此传输层路由不同于修改 lock 或重新解析依赖。镜像请求强制直连、不使用进程 HTTP(S) 代理；镜像返回 HTTP、TLS、来源跳转或哈希错误则失败，**不自动退回可能消耗 VPN 流量的官方源**。完整的内容寻址缓存直接复用；来自旧来源的未完成 `.part` 明确放弃并从镜像重新下载，同一镜像的未完成传输继续支持 Range。选择第三方镜像需要信任其可用性；SHA256 可核对固定制品的字节内容。该选项本身不会修改 PyTorch 专用仓库、Debian snapshot、OCI 或 HF 权重来源。另有独立的 `ACPROF_PYTORCH_ARTIFACT_MIRROR=aliyun`（仅锁定的 `download-r2.pytorch.org/whl/`）及 `ACPROF_DEBIAN_ARTIFACT_MIRROR=tencent`（仅固定时间戳的 `snapshot.debian.org` 包），同样保留原锁 SHA256、缓存身份、Range 恢复与镜像失败即停止的约束；默认均关闭。OCI/HF 仍单独处理。
`scripts/compile_locks.py --index-url <https-index> --variant cpu` 让固定 uv 重新从指定索引解析目标平台 artifacts；`--torch-index-url` 可显式指定相应 CUDA wheel 索引。默认保留 exact versions；同版本 artifact SHA256 改变会拒绝替换。生成 `.artifacts.json` 保存目标平台、来源与已知大小。不要在锁文件中批量替换 URL、版本或 SHA256；上述 `ACPROF_PYPI_ARTIFACT_MIRROR` 仅作受限范围、严格 SHA256 校验的传输路由。`--check --variant cpu --variant cu124 --variant cu128` 是离线锁校验，不能代替三个环境的实际构建/推理。

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
