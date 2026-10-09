# 图表与绘图入口

[← 返回专题目录](metrics.md)

## 绘图入口
```bash
acprof plot \
  results/smoke/google-bert--bert-base-uncased/result_all.csv
```

图表按环境写入绘图根目录下的 `<environment_class>/cpu/`、`<environment_class>/gpu/` 和 `<environment_class>/gpu+cpu/`，延迟模型仍写入 `latency_model/`；没有适用数据的分组会自动跳过。除原有指标总览外，还会按可用字段生成资源失败边界、P50/P90/P95 尾延迟、延迟–能耗 Pareto 前沿和冷启动阶段分解图。历史 CSV 缺少新字段时只跳过对应图，不影响其余图表。

`acprof plot` 读取实验根目录的 `static_meta.json`，用其中的 `input_scale_type` 作为横轴语义名；静态元数据要求 schema v7，旧 `static_meta.csv` 会直接报错。新实验的图片会写入结果目录的 `plots/`；没有清单的旧目录仍直接写到根部。绘图根目录下包含：

- `<environment_class>/cpu/`：只使用该环境下 `gpu_mode=off` 的 CPU 数据
- `<environment_class>/gpu/`：只使用该环境下 `gpu_mode=on` 的 GPU 数据
- `<environment_class>/gpu+cpu/`：同时包含该环境下 GPU 和 CPU 数据，用于对比

缺少环境身份的旧 CSV 归入 `unknown/`。单个输入文件混合多种环境或与元数据身份冲突时拒绝绘图，不生成跨环境排名或拟合模型。

每个有对应数据的目录按可用指标生成总览图、能耗图和专项分析图；Massif 采用独立图表，以保留其进程生命周期内存口径：

- `latency_overview_vs_scale.png`
- `service_efficiency_overview_vs_scale.png`
- `packet_overview_vs_scale.png`
- `torch_compute_overview_vs_scale.png`
- `ncu_arithmetic_overview_vs_scale.png`
- `ncu_runtime_overview_vs_scale.png`
- `nsys_timing_overview_vs_scale.png`
- `container_cpu_overview_vs_scale.png`
- `cpu_execution_overview_vs_scale.png`
- `cpu_memory_behavior_overview_vs_scale.png`
- `container_memory_process_overview_vs_scale.png`
- `container_io_overview_vs_scale.png`
- `gpu_resource_overview_vs_scale.png`
- `gpu_energy_power_overview_vs_scale.png`
- `cpu_package_energy_power_overview_vs_scale.png`
- `vcpu_estimated_energy_power_overview_vs_scale.png`
- `massif_cpu_heap_peak_total_vs_scale.png`
- `cold_start_bar.png`
- `resource_feasibility_heatmap.png`
- `tail_latency_overview_vs_scale.png`
- `latency_energy_pareto.png`
- `cold_start_breakdown.png`

13 张通用总览图统一使用 input scale 横轴、配置颜色和共享图例；每张最多 6 个子图。只有单位、语义和数值尺度都适合直接比较的子图才共享纵轴，例如 packet/application latency、NCU application/packet MFLOPS 以及 cache/dTLB 的同类比率。cache miss 和 dTLB miss 的单 request 计数可能相差多个数量级，因此使用独立纵轴。某个指标没有数据时，对应位置显示 `No data`；整张总览图的全部指标都没有数据时才跳过该 PNG。延迟总览使用 `3×2` 布局，同时展示 packet/application 的原始延迟、归一化延迟和 CV；service efficiency 总览集中展示吞吐、每 CPU core 吞吐、container-attributed energy、归一化能耗和 samples/J。分辨率类型的归一化子图使用每百万像素指标（Diffusion 为输出像素，CV/多模态为输入像素），其他尺度仍使用原来的每 input unit 指标。分辨率图缺少可靠像素数时显示 `No data`，不退回边长分母。

通用总览图和 energy/power 总览图的 `Configuration` 图例按运行模式、CPU 核数分列：`GPU+CPU1` 表示启用 GPU、CPU 配额为 1 核，`CPU1` 表示仅使用 CPU、配额为 1 核；列内的 `Mem2`、`Mem4` 等对应 `mem_cap_gb`，按数值从小到大排列并对齐。GPU 组在前，CPU-only 组在后，各组 CPU 核数递增；本图没有有效数据的配置留空，不生成额外曲线。宽图最多并排 8 组，单列子图最多并排 4 组，更多配置整组换行；画布按实际图例高度预留空间，避免遮挡标题、idle 说明或子图。曲线颜色和聚合口径保持原有规则。

三张 energy/power 总览图分别对应 GPU board、CPU package 和 estimated vCPU-attributed 口径。每张 PNG 使用 `3×2` 子图：三行依次为 energy/request、average power 和 peak power，左列展示扣除 idle baseline 的 effective 指标，右列展示保留 idle baseline 的 total 指标；同行共享纵轴，所有子图共享 input-scale 横轴、配置颜色与图例。图例下方的灰色信息框同时给出 idle 平均功率和最大的 case 内相对极差 `(max-min)/mean`；CPU package 按 CPU-only / GPU-enabled 分开，estimated vCPU 则标明由 CPU package baseline 按 interval CPU share 归因，避免把估算值误解为独立实测。这样可以直接观察 idle 对各指标的影响，不再单独生成 effective 或 total PNG。CPU package 来自 RAPL，estimated vCPU 则按 container cgroup CPU share 估算，两者不能混作同一测量口径。

`cpu_memory_behavior_overview_vs_scale.png` 使用 `2×2` 布局汇总 Linux `perf` generic PMU event，只表示 cache / dTLB miss 的单 request 计数和 miss rate，用于观察访存局部性及地址转换开销；它们不是 DRAM read/write traffic，也不是实际内存带宽 GB/s。不同 CPU 架构、虚拟化环境或 kernel PMU 可能不提供相同事件；部分字段不可用时仍会保留其余可用子图。

`container_io_overview_vs_scale.png` 的 block read/write operation 数可能相差多个数量级，因此两个子图分别按各自数据自动缩放纵轴，并各自标注 `Operations/request`。比较读写数量时应读取刻度值，不能直接比较曲线高度；零值和原始聚合数值保持不变。

`container_memory_process_overview_vs_scale.png` 中 GPU 曲线统一使用绿色，按本图 GPU 配置的 `mem_cap_gb` 从小到大由浅变深；相同 memory cap 的不同 CPU 配置同色。六个子图和配置图例使用同一映射，适用于 `gpu/` 和 `gpu+cpu/`。CPU-only 曲线沿用 CPU 色相与 memory cap 深浅，每个资源配置仍独立聚合和绘制。

Massif 图使用 `cpu_heap_peak_total_bytes_massif / 1024^3` 得到绘图期派生列 `cpu_heap_peak_total_gib_massif`；不会改写原始 CSV。它衡量进程生命周期内的 heap peak，与 container cgroup 测量窗口指标口径不同，因此保留为独立 PNG。`nsys_timing_overview_vs_scale.png` 使用 `2×2` 布局展示 host wall、CUDA API sum、GPU kernel sum 和 GPU memcpy sum。未启用对应 probe、字段不存在或整组字段均为 `nan` 时，这些可选图表会自动跳过。

四张论文分析图使用独立口径：

- `resource_feasibility_heatmap.png` 是唯一保留 `status=warn/error` 行的图。每个 `GPU mode × CPU × memory × input scale` 单元格把正式测量重复折叠为 `OK`、warning、partial failure、timeout、startup/runtime OOM、timeout 后跳过、其他错误或未知。只要同一单元格同时出现成功与失败，就标记为 partial failure，不会被成功行掩盖。
- `tail_latency_overview_vs_scale.png` 为每个 GPU mode 和 CPU 数选择有成功数据的最大 memory cap，先对 repeat window 的 P50/P90/P95 取中位数，再绘制 P50–P95 区间和 `P95/P50`。它同时展示 packet/application 口径，但不会用窗口分位数伪造 request-level violin distribution。
- `latency_energy_pareto.png` 按 input scale 分面并在 log-log 坐标中标出同时最小化 latency 与 container-attributed effective energy 的非支配前沿。延迟列依次优先使用 application P95、packet P95、application mean、packet mean；同一面板不会混合不同 input scale。历史 CSV 没有 `container_attributed_energy_eff_j` 时，只在 source fields 可用的行按现有口径重建：CPU-only 使用 estimated vCPU effective energy，GPU 行使用 estimated vCPU 与 GPU effective energy 之和。
- `cold_start_breakdown.png` 仅在五个阶段字段完整时生成，并为每个 GPU mode/CPU 数选择最大 memory cap。同一张 PNG 使用上下两个子图，共享配置横轴、独立缩放纵轴：上图堆叠 container launch、server setup、CUDA init、model load 和 ready wait，并用 `cold_start_s` 独立标记核对阶段和；下图单独展示 first-predict application latency，不计入 `/ready` 前的堆叠总量，缺少该指标时显示 `No data`。旧结果缺少阶段列时继续保留 `cold_start_bar.png`，并自动跳过分解图。

延迟建模产物统一写入绘图根目录下的 `latency_model/`（v2 为 `plots/latency_model/`）：

报告 JSON 保留源实验的 platform、collection_tier、comparability_class 与 environment_class；
residual CSV 每行写入 environment_class。缺少源身份时标记 unknown，混合环境输入拒绝拟合。

- `latency_model/latency_model_report.json`
- `latency_model/latency_model_residuals.csv`
- `latency_model/latency_model_fit_curves.png`
- `latency_model/latency_model_residuals.png`

建模前会先按 `GPU mode × CPU × memory × input scale` 对正式测量重复取中位数，确保同一个 case 的重复不会被拆到训练与测试两侧。CPU-off 使用对数空间二次响应面；GPU-on 使用连续分段 log-linear 主模型，并为不稳定的上边界配置连续 affine latency tail：

```text
CPU: latency_s = exp(
  intercept + log(input_scale) + log(input_scale)²
  + log(cpu_cores) + log(cpu_cores)² + log(mem_cap_gb)
  + log(input_scale) × log(cpu_cores)
  + log(input_scale) × log(mem_cap_gb)
  + log(cpu_cores) × log(mem_cap_gb)
)

GPU within training range: latency_s = exp(
  intercept + log(input_scale)
  + Σ hinge_k × max(0, log(input_scale) - log(k))
  + 1/cpu_cores + 1/cpu_cores²
  + log(mem_cap_gb) + log(input_scale) × 1/cpu_cores
  + log(input_scale) × 1/cpu_cores²
)

GPU activated upper tail:
  latency_s(x) = spline_latency_s(x_max)
    + affine_tail(x) - affine_tail(x_max)
```

CPU 模型除共同的二次 log input-scale 项外，还使用二次 log CPU 项及 input-scale/CPU/memory 两两交互，以表达 CPU 饱和、memory cap 影响随规模变化等非线性资源响应。GPU 延迟可能因 kernel、attention 实现或内存执行区间切换而在相邻输入规模间改变斜率，因此 GPU 模型把每个内部实测 input scale `k` 作为共享的 log-space 线性样条结点；结点之间连续插值。若嵌套的一步前向检查 MAPE 超过 5%、训练尺度跨度至少为 10 倍，且 affine tail 在全部训练资源配置上的斜率为正，则上边界外推改用与样条边界连续的 latency-space affine tail，避免把单个不稳定末段斜率无限延长。否则继续使用 log-log 样条边界段。GPU 模型仍使用一阶和二阶逆 CPU 特征，以表达主机侧开销随 CPU 增加快速下降、随后进入 GPU 主导平台区的形状。所有输出都经过正值路径或回退到指数链接；CPU/GPU 独立系数等价于在联合模型中加入 GPU 相关交互。报告分别执行两种组外验证：

- resource configuration holdout：逐次完整留出一个 `(cpu_cores, mem_cap_gb)` 配置及其全部 input scale；
- input scale holdout：仅使用较小尺度训练，完整留出最大 input scale，检验向前外推；至少需要 3 个尺度，保证留出最大值后训练侧仍有 2 个不同尺度。

报告逐 CPU/GPU 模型给出 R²、MAE、RMSE、relative MAE、MAPE、SMAPE、非正预测数、系数、数值秩和训练范围。resource configuration holdout 要求 `R² >= 0.80`；两种验证都要求总体 `relative MAE <= 0.20`、总体 `MAPE <= 0.20`、任一留出资源配置的 `relative MAE` 与 `MAPE <= 0.30`、任一验证 case 的相对误差 `<= 0.30`，且预测有限为正。input scale holdout 的测试行全部处于同一个尺度，其 R² 只衡量该固定尺度内很小的资源配置差异，不能衡量尺度水位外推是否准确，因此仍在报告中保留但不作为质量门槛。全部适用门槛通过时顶层才写入 `status=ok` 和 `prediction_ready=true`；否则使用 `poor_fit`、`unvalidated` 或 `skipped`，不会把“求解成功”误报为“可用于预测”。
