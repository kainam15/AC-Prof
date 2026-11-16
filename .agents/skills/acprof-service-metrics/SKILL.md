---
name: acprof-service-metrics
description: 设计、修改或审查 AC-Prof LLM 服务的 Prometheus 指标消费、直方图差值和 SLO Goodput；不用于部署监控平台或自动发起性能实验。
metadata:
  reference: wshobson/agents
  revision: 9c61ab243def98f72da12abd41cfcf5bcd91af9d
---

# AC-Prof 服务指标与 SLO

先读[流式指标约定](../../../docs/profiling/streaming.md#指标约定)、[SLO Goodput](../../../docs/profiling/streaming.md#slo-goodput-与单点容量)及[服务端阶段统计](../../../docs/profiling/streaming.md#服务端阶段统计)。采集字段变化还须使用[Schema 变更流程](../acprof-schema-change/SKILL.md)，诊断已有实验则使用[结果审计](../acprof-result-audit/SKILL.md)。

1. 明确指标产生者、counter/gauge/histogram 类型、单位、labels、服务身份、起止采样及适用窗口。共用现有 parser 和协议，不另建 exporter 或常驻采样器。
2. 检查 counter 与 bucket 的差值、单调性、reset、缺失、标签变化和非法值；无法归属或计算时保留原因，不能补零或裁剪负数让结果通过。
3. 严格区分客户端 TTFT/TPOT/ITL/E2E 与服务端 Queue/Prefill/Decode。共享服务累计直方图不是本次请求精确计时，共享 GPU 能耗不能直接分摊成精确 Token 能耗。
4. SLO 的阈值、成功条件、分母、有效 Token 和时间窗沿用项目声明。区分请求延迟分位数、服务端 bucket 估计与 Goodput，缺少证据时不构造总百分比。
5. 以小型 Prometheus 文本、确定性时间和合成请求 Fixture 覆盖正常、reset、缺项及身份冲突，核对 CSV、JSON、HTML 消费与来源。正式测量窗口不增加调试 I/O 或刷新。
6. 报告离线解析、真实服务端观测及 GPU/服务 sweep 的各自证据。真实端点采集、负载实验、Prometheus/Grafana 部署不因使用本 skill 自动启动。

借鉴 [prometheus-configuration](https://github.com/wshobson/agents/tree/9c61ab243def98f72da12abd41cfcf5bcd91af9d/plugins/observability-monitoring/skills/prometheus-configuration) 和 [slo-implementation](https://github.com/wshobson/agents/tree/9c61ab243def98f72da12abd41cfcf5bcd91af9d/plugins/observability-monitoring/skills/slo-implementation) 的指标类型与 SLI/SLO 区分；不复制部署模板或长期可用性预算来替代 AC-Prof 测量协议。
