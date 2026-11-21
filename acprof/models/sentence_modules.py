"""Static class references and component files used by the sentence encoder loader."""
from __future__ import annotations

from pathlib import PurePosixPath

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

# Required companion file and whether the loader calls Module.load_torch_weights.
_MODULE_FILES = {
    "Pooling": ("config.json", False),
    "WordWeights": ("config.json", False),
    "BoW": ("config.json", False),
    "Dense": ("config.json", True),
    "WeightedLayerPooling": ("config.json", True),
    "WordEmbeddings": ("wordembedding_config.json", True),
    "LSTM": ("lstm_config.json", True),
    "CNN": ("cnn_config.json", True),
    "StaticEmbedding": ("tokenizer.json", True),
}


def sentence_module_name(reference: object) -> str | None:
    return _MODULE_NAMES.get(reference) if isinstance(reference, str) else None


def module_prefix(path: object) -> str:
    if not isinstance(path, str):
        raise ModelFilesError(f"invalid model file path in modules.json: {path!r}")
    # modules.json declares directories; discovery joins metadata paths with PurePosixPath.
    directory = str(PurePosixPath(path))
    return "" if directory == "." else safe_path(directory) + "/"


def sentence_module_checkpoint(kind: str, names: set[str], prefix: str) -> Checkpoint | None:
    """Check declared files without treating config-only modules as weight checkpoints."""
    rule = _MODULE_FILES.get(kind)
    if rule is None:
        return None
    required, loads_weights = rule
    if prefix + required not in names:
        raise ModelFilesError(f"missing {kind} file: {prefix}{required}")
    if not loads_weights:
        return None
    for filename, extension in (("model.safetensors", "safetensors"), ("pytorch_model.bin", "bin")):
        if prefix + filename in names:
            return {"component": prefix.rstrip("/") or ".", "format": extension, "variant": None,
                    "files": [prefix + filename]}
    raise ModelFilesError(f"missing {kind} weights: expected {prefix}model.safetensors or {prefix}pytorch_model.bin")
