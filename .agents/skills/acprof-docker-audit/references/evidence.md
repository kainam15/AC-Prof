# 镜像取证分支

只读取与当前问题有关的分支。镜像类型、空间算法与删除规则由[镜像管理文档](../../../../docs/models/images.md#镜像管理与清理)维护。

## 身份与连接

在任务指定的 Docker 连接上执行；已有显式 context 或连接参数时保持一致，不切换全局默认值。

```bash
docker context show
docker info --format '{{json .ID}}'
docker image ls --no-trunc
docker image inspect --format '{{json .Id}} {{json .RepoTags}} {{json .RootFS.Layers}}' IMAGE
docker ps -a --no-trunc --format '{{.ID}}\t{{.Image}}\t{{.Status}}'
```

将 `IMAGE` 换为用户指定的完整引用或 ID。只提取需要的 AC-Prof label；不输出完整容器环境或认证信息。记录实际连接来源，`context show` 不能独自证明环境变量或显式 host 参数没有覆盖连接。

| 观察 | 下一步 | 可得结论 |
| --- | --- | --- |
| 多个引用指向同一完整 ID | 汇总全部 RepoTags | 引用别名，不是重复存储的多份镜像 |
| 名称相同但 ID 不同 | 比较父层、Diff ID 链和身份标签 | 区分服务代码层差异与权重差异 |
| TUI 名称没有 Docker tag | 查详情或当前映射代码 | 可能是逻辑节点，不据此构造删除命令 |
| 列表有镜像却触发重建 | 比较本轮与已有镜像的完整构建输入 | 解释复用条件不匹配的具体字段 |

## 为什么重建

沿[构建、复用和验证](../../../../docs/models/images.md#构建复用和验证)读取当前规则，按需要检查模型及依赖 revision、下载计划、平台、环境锁、adapter/backend 和进入构建身份的项目文件。

区分镜像查询、拉取认证、网络传输和构建命令失败。不要把历史版本的标签前缀、层数或指纹字段清单当成永久契约。用本次日志说明复用了哪层、重建了哪层。

## 空间估算

```bash
docker system df -v
docker buildx ls
docker buildx du --builder BUILDER
```

`BUILDER` 使用实际构建该批镜像的 builder。未查明时列为证据缺口，不用默认 builder 的缓存解释另一 builder。

分别报告镜像共享/独占层、构建缓存和文件系统可用量。远程 Docker 的存储路径不能在客户端执行 `df` 后冒充服务端结果；无法证明对应关系时记为未知。空间计算的详细口径见权威文档，保留原始单位与字节数。

历史案例的可复用判断：两个 MOSS 服务镜像可以共享大部分模型层；删除一个标签未必删除镜像；删除镜像后 BuildKit 仍可能保留对应数据。不要复用历史 ID、大小或“可删除”判断。

## 上游依据

[Docker buildx du](https://docs.docker.com/reference/cli/docker/buildx/du/)与[镜像 inspect](https://docs.docker.com/reference/cli/docker/image/inspect/)用于核对当前 CLI 行为；只有版本差异或语义不明时再读取。沿用 Docker 工具，不引入第二套镜像扫描器。
