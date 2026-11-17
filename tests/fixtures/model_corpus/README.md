# Offline model corpus fixtures

[`../model_corpus.json`](../model_corpus.json) declares 17 reviewed synthetic
scenarios for the real model-inspection and coverage pipeline. No weights are
included or downloaded. A listed artifact name tests a layout, not its contents.
The repeated `aaaaaaaa…` revision is a synthetic identity, not a real Hub commit.

The T5 case reuses the original files in
[`../t5_small_metadata/`](../t5_small_metadata/) at the pinned commit and SHA256
recorded in the manifest. Its Hub observation and repository file list are still
synthetic, so the whole case is not a captured repository snapshot. The author's
summarization and translation parameters form two inference groups and require
selection under the existing resolver rules.

Cases cover native NLP, architecture-only detection, ModelScope source identity,
timm vision, Whisper audio, multimodal generation, DDPM diffusion and declared
ONNX classification. Negative cases preserve unsupported GGUF, unknown artifacts,
unknown architectures, conflicting Hub hints, missing ONNX configuration,
missing shards, invalid metadata and missing tokenizer dependencies.
`invalid-metadata/config.json.txt` intentionally contains non-finite JSON input;
it is replayed as repository `config.json` to test the production diagnostic.

When adding or changing a case:

1. Keep source, model ID, revision and `case_id` explicit and stable. Do not label
   constructed observations as captured metadata.
2. Store exact small metadata bytes and their SHA256. Captured files need pinned
   source URLs; a captured repository observation also needs `capture_source`.
3. Review expected status, reason and relevant task/profile against the intended
   contract. Do not automatically bless the current resolver output.
4. Replay and inspect the original attempts and differences. Comparisons require
   identical inputs; intentional input changes start a new baseline. Changing
   expectations alone retains input identity and remains visible separately.

Commands, output schema, evidence boundaries and design references are documented
in [offline metadata regression](../../../docs/usage/cli-discovery.md#离线元数据回归).
