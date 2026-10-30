"""Pure input-construction bounds shared by host generators and container handlers.

These limits bound expanded input objects, not model memory or OOM feasibility.
Explicit workload overrides travel with each frozen payload, never via host RAM.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from acprof.experiments.constraints import ConstraintViolation, integer


class InputBudgetExceeded(ValueError):
    """Construction was refused before allocation; no inference OOM occurred."""


def positive_integer(value: object, name: str, *, allow_integral_float: bool = True) -> int:
    """Parse numeric integers without truncation; strict fields reject floats."""
    try:
        parsed = integer(value, allow_integral_float=allow_integral_float)
    except ConstraintViolation as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class InputBudget:
    max_batch_items: int = 4096
    max_text_characters: int = 16_777_216
    max_pixels: int = 67_108_864

    @classmethod
    def from_value(cls, value: object = None) -> "InputBudget":
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("input_budget must contain only max_batch_items, max_text_characters, max_pixels")
        if any(type(limit) is not int or limit < 1 for limit in value.values()):
            raise ValueError("input_budget limits must be positive integers")
        return cls(**value)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @staticmethod
    def _check(name: str, amount: int, limit: int) -> None:
        if amount > limit:
            raise InputBudgetExceeded(f"input_budget_exceeded: {name}={amount} exceeds {limit}")

    def batch(self, count: int) -> None:
        self._check("batch_items", positive_integer(count, "batch_size"), self.max_batch_items)

    def text(self, characters: int, *, batch_items: int) -> None:
        self.batch(batch_items)
        self._check("text_characters", characters, self.max_text_characters)

    def image_collection(self, sizes: Iterable[tuple[int, int]], *, batch: int = 1) -> None:
        """Bound a heterogeneous request, including every ordered video frame."""
        batch = positive_integer(batch, "batch_size")
        total = 0
        for count, (width, height) in enumerate(sizes, 1):
            self.batch(count * batch)
            total += positive_integer(width, "width") * positive_integer(height, "height")
            self._check("pixels", total * batch, self.max_pixels)

    def pixels(self, width: int, height: int, *, frames: int = 1, batch: int = 1) -> None:
        width, height = positive_integer(width, "width"), positive_integer(height, "height")
        frames, batch = positive_integer(frames, "num_frames"), positive_integer(batch, "batch_size")
        self.batch(batch * frames)
        self._check("pixels", width * height * frames * batch, self.max_pixels)


def check_nlp_input(raw: dict[str, Any], task: str, *, embedding_prompt: str = "") -> InputBudget:
    """Count request expansion without allocating samples or invoking a tokenizer."""
    budget = InputBudget.from_value(raw.get("input_budget"))
    batch = positive_integer(raw.get("batch_size", 1), "batch_size")
    budget.batch(batch)

    def length(value: object) -> int:
        return len(value) if isinstance(value, str) else 0

    items = 1
    characters = length(raw.get("text")) + length(embedding_prompt)
    if task in {"sentence-similarity", "text-ranking"}:
        documents = raw.get("documents", [])
        if isinstance(documents, list):
            items = len(documents) + (1 if task == "sentence-similarity" else 0)
            budget.batch(max(1, items) * batch)
            characters = sum(length(doc) for doc in documents)
            characters += length(raw.get("query")) * (1 if task == "sentence-similarity" else len(documents))
            characters += length(embedding_prompt) * items
    elif task == "zero-shot-classification":
        labels = raw.get("candidate_labels", [])
        if isinstance(labels, list):
            items = len(labels)
            budget.batch(max(1, items) * batch)
            template = raw.get("hypothesis_template", "This example is {}.")
            # Formatting is bounded before building any hypothesis strings.
            template_size = length(template)
            slots = template.count("{}") if isinstance(template, str) else 0
            characters = (characters + template_size) * items + slots * sum(length(label) for label in labels)
    elif task == "question-answering":
        characters = length(raw.get("question", raw.get("text"))) + length(raw.get("context", raw.get("text")))
    elif task == "table-question-answering":
        table = raw.get("table", {})
        if isinstance(table, dict):
            characters = length(raw.get("query")) + sum(
                max(1, length(cell)) for column in table.values() if isinstance(column, list) for cell in column
            )
    budget.text(characters * batch, batch_items=max(1, items) * batch)
    return budget
