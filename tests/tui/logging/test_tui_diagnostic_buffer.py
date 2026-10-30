"""Diagnostic retention limits are observable, including total omission."""
import pytest

from acprof.tui.logging.diagnostic_buffer import BoundedLogBuffer


def test_keeps_first_context_and_latest_tail_with_exact_discard_count():
    buffer = BoundedLogBuffer(max_entries=4)
    for text in ("first", "a", "b", "c", "last"):
        buffer.append(text)
    batch = buffer.drain()
    assert batch.entries == ("first", "b", "c", "last")
    assert batch.discarded_entries == 1
    assert batch.oversized_entries == 0
    assert batch.retained_bytes == len(b"firstbclast")
    assert len(buffer) == buffer.byte_size == buffer.discarded_entries == 0
    assert not buffer


def test_utf8_payload_limit_is_not_a_character_limit():
    buffer = BoundedLogBuffer(max_entries=4, max_bytes=12, max_entry_bytes=12)
    for text in ("你", "好好", "呀呀"):
        buffer.append(text)
    batch = buffer.drain()
    assert batch.entries == ("你", "呀呀")
    assert batch.retained_bytes == 9
    assert batch.discarded_entries == 1


def test_all_oversized_records_still_produce_an_omission_batch():
    buffer = BoundedLogBuffer(max_entry_bytes=4)
    buffer.append("汉字")
    buffer.append("large")
    assert buffer
    batch = buffer.drain()
    assert batch.entries == ()
    assert batch.retained_bytes == 0
    assert batch.discarded_entries == batch.oversized_entries == 2
    assert not buffer


def test_huge_records_are_rejected_before_encoding_and_clear_resets_budget():
    class Huge(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("unbounded encoding")
    buffer = BoundedLogBuffer(max_entries=1, max_bytes=4, max_entry_bytes=4)
    buffer.append(Huge("x" * 100))
    buffer.append("abcd")
    buffer.append("")
    assert buffer.drain().entries == ("",)
    buffer.append("last")
    buffer.clear()
    assert not buffer
    assert buffer.byte_size == buffer.discarded_entries == buffer.oversized_entries == 0


@pytest.mark.parametrize("bound", ["max_entries", "max_bytes", "max_entry_bytes"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_limits_are_rejected(bound, invalid):
    with pytest.raises(ValueError, match="positive integer"):
        BoundedLogBuffer(**{bound: invalid})
