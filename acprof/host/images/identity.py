"""Stable model token shared by container, image and payload names."""


def model_token(model_id: str) -> str:
    """生成镜像、容器和文件名的模型标识；转小写、展开斜线并保留点号。"""
    return model_id.replace("/", "--").lower()
