# 测量容器出站隔离

## 目标与当前边界

目标是阻断模型容器主动访问公网、局域网、其他容器和宿主服务，同时保留宿主发起的
`/ready`、`/predict` 请求及其响应。容器内部 `127.0.0.1` 和 `::1` 通信可以保留。
下载和镜像准备在正式测量之前完成。

当前 DockerRuntime 仍使用 bridge 和模型离线环境变量，**尚未启用出站封锁**。
本专题定义独立验收方案，`scripts/validate_measurement_egress.py` 不改变正式运行配置。
实验结果不能直接视为完整采集、cold start 或所有模型的性能验收。

## 方案选择

| 候选 | 测量链路与限制 | 结论 |
| --- | --- | --- |
| `--network none` | 不能保留现有宿主 HTTP 入口 | 不采用 |
| internal bridge | 隔离外部路由，但宿主网关服务仍可达；端口发布须独立核对 | 不能满足阻断宿主服务的目标 |
| 容器 network namespace 内的 nftables output 策略 | 保留 bridge、回环端口发布与 `docker0` 抓包；在容器应用启动前安装规则 | 选为后续正式接入方案，仍需完成下述接入验收 |

策略使用 `inet` 表覆盖 IPv4/IPv6，默认拒绝输出，仅允许目标为容器本地
`127.0.0.1` / `::1` 的 loopback 通信，以及 conntrack 中 `established,related`
且方向为 `reply` 的响应。仅允许 `established` 而不限定方向，会放行容器此前主动建立的连接。
Docker 的 `127.0.0.11` DNS 代理不在 loopback 例外内，避免将宿主 DNS 转发作为外联通道。
规则源唯一保存在 `scripts/egress_namespace.py`，不在宿主 namespace 写规则。

helper 检查容器 PID 的启动时间，固定 namespace 文件描述符，拒绝当前宿主 namespace，
再进入目标 namespace 安装规则。容器保持 `--cap-drop=ALL` 和 `no-new-privileges`，
应用不能移除规则。规则随容器 namespace 释放；不修改宿主全局 firewall。
该方案不防御宿主管理员、Docker 控制权限或内核漏洞。

## 独立验收

先准备已缓存 `hf-internal-testing/tiny-random-bert` 权重的 AC-Prof fill-mask 服务镜像。
验收使用既有不可变镜像和只读 Model Store，不下载或构建镜像。
需要 Native Linux、Docker、NVIDIA GPU、nftables、tcpdump、tshark，以及可执行
namespace helper / tcpdump 的 `sudo -n`。脚本持有全局测量锁；失败时不绕过锁。

从完整 Git checkout 的根目录执行（旧解析器比较使用仓库中的 `acc6345` 提交）：

```bash
uv run --locked python scripts/validate_measurement_egress.py \
  --image '<existing-immutable-image-id>' \
  --gpu-uuid '<GPU-UUID>' \
  --output internal-testing/egress-acceptance
```

输出目录必须不存在。脚本记录镜像身份、模型 revision、代码版本及 helper/script 哈希、
系统与工具版本、实际命令、原始抓包、响应耗时、NVML telemetry、规则计数器与清理结果。
测试容器和 network 使用专属标签，只回收本次创建的资源。

- 对比 internal bridge 的宿主直连、端口发布与主动外联。
- 分别运行 CPU/GPU 的 bridge 对照和受控出站服务，验证 `/ready` 返回的设备及 `/predict`。
- 仅在请求窗口之外检查宿主、同网段容器和公网 TCP；额外验证 UDP、Docker DNS 和 IPv6。
- 保留回环端口映射与 `docker0` 抓包，每组 12 个请求；逐字段比较当前与旧版 pcap 解析结果。
- 策略安装在入口 gate 放行前完成；安装失败时停止容器，不启动模型应用。
- 退出时先清理容器/network，再释放 Model Store lease；保存失败和清理错误。

内核缺少 IPv6 时，结果明确记录 `ipv6_protocols.status=unavailable`，不能声称 IPv6 通过。
公网对照本身超时也不能作为策略有效的证据；须结合可达的宿主/同网段 canary、受控请求的
拒绝结果和规则计数器判断，并保留公网基线限制。`--protocols-only` 可单独复验网络协议。

## 正式接入前的验收条件

独立实验的启动耗时包含容器创建、入口 gate、策略安装与 ready 等待，不是正式 cold-start 字段。
后续接入 DockerRuntime 须将策略身份纳入运行 provenance，明确特权 helper 的调用与失败边界，
在每个真实启动、异常退出和恢复路径验证清理，且不得在正式请求窗口内插入探针或规则查询。
GPU profiler 等独立工具容器需要单独评估，不继承普通模型服务的验收结论。

CPU/GPU 实际 HTTP 链路、端口发布、抓包字段、IPv4/IPv6 阻断以及正式延迟窗口均应分别验收。
只有这些条件完成后才能将运行环境描述为已启用出站隔离。

## 参考依据

- [Docker network create](https://docs.docker.com/reference/cli/docker/network/create/#internal)：internal network 的宿主网关边界。
- [Docker port publishing](https://docs.docker.com/engine/network/port-publishing/)：bridge gateway mode 与端口发布行为。
- [Moby DNS forwarding advisory](https://github.com/moby/moby/security/advisories/GHSA-mq39-4gv4-mvpx)：internal network 与 DNS 转发边界，说明为何单独验收 DNS。
- [nftables conntrack 匹配](https://wiki.nftables.org/wiki-nftables/index.php/Matching_connection_tracking_stateful_metainformation)：连接状态和原始/响应方向。

只借鉴官方设计与接口；不复制上游实现、不增加 Python 依赖，不在正式测量期启动附加守护进程。
