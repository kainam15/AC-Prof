# Linux Runtime 与受限环境

[文档导航](../README.md) · [CLI 参数](../usage/cli-run.md) · [TUI](../usage/tui.md)

AC-Prof 根据实际可访问的接口选择执行与采集路径。云厂商名称不作为硬件权限或资源隔离的证据。
Native Linux 的 Docker `full` 仍是完整测量基准；Process 当前提供 `basic`。

## 选择与准备

`--runtime auto|docker|process` 的默认值为 `auto`：

1. Docker CLI、本机 daemon 访问和统一 cgroup v2 检查通过时选择 Docker，保留原有矩阵和测量协议。
2. Docker 不可用、系统为 Linux 且请求 `basic` 时选择 Process，输出选择原因。
3. 显式 Docker 或 `full` 的前提缺失即失败。开始模型准备后不切换 Runtime；推理失败也不触发自动重试或降级。

Process 要求 Linux x86_64、可用的 `uv`、procfs，以及允许 pidfd 和 child-subreaper 的内核接口
（pidfd 通常需要 Linux 5.3+；seccomp 策略也可能禁止）。无需 Docker CLI、socket、root 或 systemd。
CLI、doctor 和 TUI 在依赖准备前通过短生命周期 helper 验证进程所有权接口，失败时明确阻止启动。
当前依赖锁尚未覆盖 ARM。系统动态库由主机提供，AC-Prof 不自动使用 apt 或 sudo 安装系统依赖。

准备阶段按现有 runtime profile 的完整 wheel 哈希锁创建独立 Python 环境，复制服务代码，
固定模型 revision，并复用 ModelStore 下载与校验。随后执行接口检查、输入规划及每个请求设备上的真实推理验证。
安装依赖和完整下载模型前，Process 与 Docker 共用[下载预检与确认](../models/model-store.md)：
模型先 HEAD 和限量 GET，依赖按实际 uv lock URL 做来源检查；取消下载确认不会运行安装器。
缓存默认位于 `~/.cache/acprof/process-environments`，可通过 `ACPROF_PROCESS_STORE` 调整；
依赖版本、已安装文件树、服务代码、Python 解释器及模型身份参与复用校验；版本号未变的依赖代码修改也会拒绝复用。
文件树哈希只在准备/恢复阶段计算，不进入测量窗口。
测量期间使用已准备的环境和 snapshot，不安装包或下载模型。HF offline 设置不等于网络隔离；Python 环境也不是安全沙箱。

```bash
acprof doctor --runtime process --profiling-mode basic --gpus off
acprof tui
```

在 TUI 选择 Smoke/basic，再到高级参数选择 Process，点击“使用 Process 继承限制”。
环境检查展示可用采集器及缺失状态；最小内存探测仍为 Docker 工具，Process 使用输入规模扫描。
Process 的结构化 `case_selected` 进度携带 `runtime_kind=process`，允许 CPU/内存为 null，界面显示“继承限制”；
Docker 与最小内存 probe 的进度仍要求正数资源轴。
CLI 的小型 ONNX 示例（从仓库根目录运行）：

```bash
uv run acprof run --runtime process --profiling-mode basic \
  --model Ritual-Net/iris-classification \
  --revision 273d867c65e8e5fa5c64c23dcf19738282452beb \
  --task tabular-classification --backend onnxruntime \
  --model-spec examples/onnxruntime/iris.model.json \
  --workload-spec examples/onnxruntime/iris.json \
  --cpus auto --mems auto --gpus off \
  --warmup 0 --repeat 1 --repeat-in-window 2 \
  --idle-seconds 0 --idle-cooldown-seconds 0 \
  --output-dir results/process-smoke --notify none
```

## 资源与指标口径

| 项目 | Process 行为 |
| --- | --- |
| `--cpus auto --mems auto` | 继承当前容器/VM 的限制；结果中的数值资源轴为空，case ID 使用 `inherited`，不会记录成零或已实施配额 |
| 数值 CPU / 内存矩阵 | 仅在可写的 cgroup v2 委托中创建本次子组、写入配额、验证进程成员和实际限制后运行；不能超过可见上级限制 |
| `--cpuset-cpus` | 启动前设置 affinity，核对实际生效集合；不代表 CPU 独占或等价 CPU 时间配额 |
| CPU 时间与利用率 | psutil 观测主服务进程及可见后代；同一进程身份的 user+system CPU 时间差，一核满载为 100% |
| 内存 | 观测进程树 RSS 总和的时间加权均值与采样峰值；共享页可能重复计算，不等价于 cgroup memory usage |
| GPU | 尊重 `CUDA_VISIBLE_DEVICES`；`--gpu-device` 数字为可见序号。保存 CUDA UUID、原始可见设备与 NVML 查询证据 |
| GPU 利用率 / 显存 | 分别查询；任一失败不抹掉另一项。整卡指标标记 `device`、共享状态 `unknown`，不声称模型独占 |
| 启动 | `process_startup_s` 记录启动到通过带随机 token 的 `/ready`；Docker `cold_start_*` 保持缺失，两个口径不能替换 |

`gpu_device_uuid` 仍表示选定设备的身份：Docker 使用现有 device request 选择，Process 使用 CUDA 可见序号解析，
具体来源保存在 `gpu_device` 元数据中；共享字段的登记来源为 `selected_device_identity`，数值与归属范围没有替换。

数值矩阵使用当前 cgroup 下的委托，或显式 `ACPROF_CGROUP_ROOT` 指向的 cgroup v2 委托。
AC-Prof 不自动修改上级 controller 或主机服务权限。只有输入规模扫描不需要额外资源控制权限。
可见的 CPU affinity、各级 quota 和 memory 上限保存在分配证据中；隐藏的祖先限制及资源共享仍可能未知。
短命子进程可能落在采样间隔之间；已观察到子进程退出且缺少终值时，CPU 总量保留缺失证据。

能力状态区分 `unsupported`、`permission_denied`、`unavailable`、`not_requested` 和 `error`。
可选资源指标缺失不取消有效的请求延迟/吞吐；缺失数值保持 NaN/null，原因保存在字段状态、采集计划与原始证据中。
Process 不运行 RAPL、PMU、抓包或独立 profiler；显式请求这些完整测量不会静默变成 basic。
现有 `probe`、post-hoc profiler 和负载重放命令仍要求 Docker。

## 生命周期与恢复

统一 Runtime 接口拥有准备、验证、启动到就绪、停止与清理，以及执行证据。
Docker adapter 调用原有生命周期；Process 通过独立 supervisor 和控制管道管理进程树。
控制进程退出（包括 SIGKILL）会关闭管道，supervisor 收养 double-fork 后代，先 TERM 再 KILL，并保存清理结果。
临时监听 socket 的 FD 直接传给 Flask 服务，避免先选端口再绑定的竞争。超时和异常退出保留日志及退出码。
清理时按 PID 区分主工作负载和后代的退出状态；即使 supervisor 的通用回收先于 `Popen.poll()`，
主工作负载的非零退出码或负信号编号也必须保留，后代的退出码不能替代它。
仅本次 cgroup 的 `memory.events oom_kill` 增量可证明 OOM；单独的 SIGKILL 不算 OOM。

模型缓存 lease 由 supervisor 持有到清理完成。无法证明全部退出时保留清理债务并停止后续 case。
恢复校验 PID 身份及 supervisor 的完成证据，不对可能已复用的 PID 盲目发送信号。
若 supervisor 本身异常消失且没有完整清理证据，拒绝自动恢复。
恢复也校验 Runtime、依赖/模型身份、继承资源配置和 GPU 可见性；条件变化要求创建新实验。

## 结果与可比性

静态元数据、CSV 与 run state 保存 Runtime 身份；每个 case 的硬件边界记录保存实际限制证据。
带有 `runtime_kind` 的新结果在每个 CSV 分层重复该身份，读取时核对一致性；历史无此字段的分层保持原有表头。
`audit` 核对 Process 身份、数值资源轴与实施限制，拒绝把 Process 数值写进 Docker 容器/冷启动字段。
统计按 Runtime、环境和指标范围分组；合并与对照检查拒绝直接混合 Process/Docker 或不同环境。
继承上限、GPU 可见性或其他比较条件缺失时，可比性为 unknown，不推断为等价。
`report` 和 TUI 可查看 Process 指标；`plot` 输出输入规模图，不对未实施的 CPU quota 拟合资源缩放模型。

## 支持与验证边界

| 环境 | 实现路径 | 验证边界 |
| --- | --- | --- |
| Native Linux | Docker basic/full；Process basic | Process CPU 小型 ONNX 已在无 Docker CLI、不可用 Docker socket 的启动配置下完成 Smoke、audit 与退出检查；Docker 完整硬件协议仍需回归实测 |
| 普通 Linux VM / 云 VM | 本机 Docker basic 或 Process basic | 无 RAPL/PMU 不阻止 basic；AWS/阿里云/腾讯云尚需实例实测 |
| AutoDL 类无 Docker GPU 容器 | Process basic，GPU 遥测按权限 | 尚未取得对应云容器与 GPU 的端到端验收证据 |
| WSL2 | basic/PARTIAL | 保持独立环境身份，不能进入 Native 完整画像基准 |
| Kubernetes / 已有远程 Endpoint | 无专用 Runtime | 后续阶段；不以当前 Process 支持宣称已验证集群兼容 |

上述 Process Smoke 的主机仍安装 Docker；验收命令通过限制 PATH 和配置不可用 socket 隔离该入口，
不等价于 AutoDL、无 Docker 镜像或 GPU 实例实测。故障回归覆盖控制进程取消/异常退出、后代进程回收与缺失权限。
同一小型 ONNX 还已完成 Docker basic 与生产 TUI headless 的真实子进程验收；两条路径均通过结果审计。
Process 的首次准备和后续缓存复用均通过文件完整性核验。Headless 证据不代表用户 SSH/IDE 终端已验收。

## 参考与取舍

- [psutil 进程实现](https://github.com/giampaolo/psutil/blob/master/psutil/__init__.py)（BSD-3-Clause）：复用进程身份、CPU/RSS 和子树接口；采样限定为目标树，不扫描全机所有进程。
- [OpenTelemetry process scraper](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/receiver/hostmetricsreceiver/internal/scraper/processscraper/process_scraper.go)及[权限问题讨论](https://github.com/open-telemetry/opentelemetry-collector-contrib/issues/34557)（Apache-2.0）：借鉴逐项失败与部分证据，未引入 Collector 服务或依赖。
- [uv 环境管理](https://github.com/astral-sh/uv/blob/main/docs/pip/environments.md)：复用现有 uv 与依赖锁，环境准备全部在测量前完成。
- [vLLM serving benchmark](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py)（Apache-2.0）：保留客户端端到端延迟与服务端推理阶段的区别，未引入 vLLM 依赖。

这些参考用于接口和证据边界设计，未复制上游实现。DCGM、Prometheus、云厂商 SDK 和 Kubernetes Runtime 不在本阶段。
