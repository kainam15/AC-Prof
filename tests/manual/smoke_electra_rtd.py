"""Manual offline CPU smoke using a tiny randomly initialized ELECTRA head.

Run in the pinned NLP CPU image; no Hub access, weights download or
measurement/profiler claim. This checks real Transformers load and inference.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import BertTokenizerFast, ElectraConfig, ElectraForPreTraining

from acprof.container.handlers.nlp import NLPHandler
from acprof.container.validation import validate_output


def main() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "vocab.txt").write_text(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nsmall\nmodel\n",
            encoding="utf-8",
        )
        BertTokenizerFast(vocab_file=str(root / "vocab.txt"), do_lower_case=True).save_pretrained(root)
        config = ElectraConfig(
            vocab_size=9, embedding_size=16, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=64,
        )
        ElectraForPreTraining(config).save_pretrained(root)
        handler = NLPHandler()
        context = handler.load(str(root), "replaced-token-detection", "transformers_model", "cpu")
        for batch in (1, 2):
            raw = {"text": "hello world small model", "params": {}, "batch_size": batch}
            prepared = handler.preprocess(context, raw)
            raw["input_scale"] = prepared["_effective_input_scale"]
            prediction = handler.predict(context, prepared)
            response = handler.postprocess(context, prediction)
            validated = validate_output(context, raw, prepared, prediction, response)
            assert response["output_shape"] == [batch, 6]
            assert validated["task"]["status"] == "verified"
            print(f"SMOKE PASS batch={batch} logits_shape={response['output_shape']}")


if __name__ == "__main__":
    main()
