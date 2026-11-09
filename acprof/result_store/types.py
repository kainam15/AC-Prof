"""Known fields of local result protocols, validated before typed consumption."""
from typing import Literal, NotRequired, TypedDict


class ResultReference(TypedDict):
    path: str
    sha256: str


class LayerManifest(TypedDict):
    path: str
    fields: list[str]
    row_count: int
    committed_bytes: int
    sha256: str
    revision: int


class ResultManifestBody(TypedDict):
    schema_version: int
    run_id: str
    state: str
    sequence: int
    wide_fields: list[str]
    schema_sha256: str
    row_count: int
    layers: dict[str, LayerManifest]
    snapshot: NotRequired[ResultReference]


class ResultManifest(ResultManifestBody):
    commit: ResultReference


class PendingTransaction(TypedDict):
    before: ResultManifest
    after: ResultManifestBody
    replacements: list[str]


class ResultSource(TypedDict):
    kind: Literal["window", "derivation"]
    reference: ResultReference


class CommitRecord(TypedDict):
    schema_version: int
    previous: ResultReference | None
    after: ResultManifestBody
    sources: list[ResultSource]


class InitializationReceipt(TypedDict):
    run_id: str
    fields: list[str]
