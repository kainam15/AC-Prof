# Offline model corpus fixtures

[`../model_corpus.json`](../model_corpus.json) declares three captured repository
observations and 17 reviewed synthetic scenarios for the model-inspection and
coverage pipeline. No weights are included or downloaded. A listed artifact
name tests a layout, not its contents. The repeated `aaaaaaaa…` revision is a
synthetic identity, not a real Hub commit.

- Hugging Face `google-bert/bert-base-uncased` (`86b5e093`): the revision API
  supplied the Hub attributes and all repository file names. The fixture keeps
  the original model and tokenizer configs, not the ONNX or native weights.
- Hugging Face [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af)
  (`169d4a43`): the same revision as the hardware validation example, replayed
  here without running it. The revision API supplied the file list and Hub
  attributes; the four original JSON configs match its Git blob IDs. This
  exercises a speech model with its real generation settings, tokenizer and
  feature extractor, not just an architecture name.
- ModelScope `Qwen/Qwen3-0.6B` (`09b42cad`): the recursive file list and the
  model, generation and tokenizer configs are pinned to the ModelScope commit;
  their bytes matched the hashes from that listing. The ModelScope card's task,
  tags and license are repository-level observations dated in `capture_source`,
  not historical properties guaranteed by the commit.

All three upstream repositories declare [Apache-2.0](../../../LICENSE). The checked-in
configs retain the upstream bytes, including missing final newlines; do not
reformat them. `metadata` records each pinned download URL and SHA256.

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
