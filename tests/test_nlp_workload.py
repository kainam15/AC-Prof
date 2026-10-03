import pytest

from acprof.workloads.nlp import NLPWorkloadGenerator


def test_zero_shot_has_nonempty_deterministic_labels_and_batch():
    generator = NLPWorkloadGenerator("model", "zero-shot-classification", 3)
    first = generator.generate(8)
    assert (first) == (generator.generate(8))
    assert (len(first["candidate_labels"])) >= (2)
    assert (first["batch_size"]) == (3)

def test_pair_tasks_scale_candidate_length_with_fixed_query_and_count():
    for task in ("sentence-similarity", "text-ranking"):
        generator = NLPWorkloadGenerator("model", task, 2)
        small, large = generator.generate(8), generator.generate(16)
        assert (small["query"]) == (large["query"])
        assert (len(small["documents"])) == (2)
        assert ([len(x.split()) for x in small["documents"]]) == ([8, 8])
        assert ([len(x.split()) for x in large["documents"]]) == ([16, 16])
        assert (small["batch_size"]) == (2)

def test_table_qa_scales_real_rows_and_reports_row_defaults():
    generator = NLPWorkloadGenerator("model", "table-question-answering", 2)
    payload = generator.generate(8)
    assert (payload["input_scale_type"]) == ("table_rows")
    assert ({len(col) for col in payload["table"].values()}) == ({8})
    assert (all(isinstance(v, str) for col in payload["table"].values() for v in col))
    assert (generator.scale_label(8)) == ("rows8")
    assert (generator.default_input_scales()) == ([1, 2, 4, 8, 16, 32])

@pytest.mark.parametrize('scale', (0, -1, 1.5))
def test_table_qa_rejects_fractional_or_empty_row_counts(scale):
    generator = NLPWorkloadGenerator("model", "table-question-answering", 1)
    with pytest.raises(ValueError, match="positive integer"):
        generator.generate(scale)
