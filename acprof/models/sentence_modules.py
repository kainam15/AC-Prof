"""Static class references and component files used by the sentence encoder loader."""
from __future__ import annotations

from acprof.models.files import Checkpoint, ModelFilesError, safe_path

# Sentence Transformers 5.6 saves the implementation path; older snapshots use the models namespace.
_MODULE_PATHS = (
    "base.modules.transformer.Transformer",
    "base.modules.dense.Dense",
    "sentence_transformer.modules.pooling.Pooling",
    "sentence_transformer.modules.normalize.Normalize",
    "sentence_transformer.modules.weighted_layer_pooling.WeightedLayerPooling",
    "sentence_transformer.modules.word_weights.WordWeights",
    "sentence_transformer.modules.bow.BoW",
    "sentence_transformer.modules.word_embeddings.WordEmbeddings",
    "sentence_transformer.modules.lstm.LSTM",
    "sentence_transformer.modules.cnn.CNN",
    "sentence_transformer.modules.dropout.Dropout",
    "sentence_transformer.modules.static_embedding.StaticEmbedding",
)
_MODULE_NAMES: dict[str, str] = {}
for _path in _MODULE_PATHS:
    _name = _path.rsplit(".", 1)[1]
    _MODULE_NAMES[f"sentence_transformers.models.{_name}"] = _name
    _MODULE_NAMES[f"sentence_transformers.{_path}"] = _name


def sentence_module_name(reference: object) -> str | None:
    return _MODULE_NAMES.get(reference) if isinstance(reference, str) else None


def module_prefix(path: str) -> str:
    return safe_path(path) + "/" if path not in ("", ".") else ""


def dense_checkpoint(names: set[str], prefix: str) -> Checkpoint:
    if prefix + "config.json" not in names:
        raise ModelFilesError(f"missing Dense configuration: {prefix}config.json")
    for filename, extension in (("model.safetensors", "safetensors"), ("pytorch_model.bin", "bin")):
        if prefix + filename in names:
            return {"component": prefix.rstrip("/") or ".", "format": extension, "variant": None,
                    "files": [prefix + filename]}
    raise ModelFilesError(f"missing Dense weights: expected {prefix}model.safetensors or {prefix}pytorch_model.bin")
