# T5 small author metadata

Unmodified `config.json` and `generation_config.json` from
[`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small/tree/df1b051c49625cf57a3d0d8d3863ed4d13564fe4),
revision `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` (Apache-2.0).
`source.json` records the exact URLs and SHA256 digests. No weights are included.

The summary declaration sets `no_repeat_ngram_size=3`; the three translation
declarations leave it unset. AC-Prof normalizes the latter to `0`, producing two
execution groups despite their shared Beam count. Offline tests use this fixture
for discovery, presentation and matrix dispatch; they do not prove model inference.
