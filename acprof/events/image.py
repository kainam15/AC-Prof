"""Docker preparation progress; byte counts describe one identified transfer, never a whole image estimate."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ImageProgress:
    component: str
    image: str
    phase: str
    detail: str = ""
    current: int | None = None
    total: int | None = None
    content: dict | None = None
    reason_code: str = ""

    def validate(self) -> None:
        if (self.component not in {"platform", "environment", "model", "service"}
                or self.phase not in {"checking", "pulling", "fallback", "building", "verifying", "complete", "failed"}
                or not isinstance(self.image, str) or not 0 < len(self.image) <= 512
                or not isinstance(self.detail, str) or len(self.detail) > 2048
                or not isinstance(self.reason_code, str) or len(self.reason_code) > 128):
            raise ValueError("invalid Docker preparation progress")
        if (self.current is None) != (self.total is None):
            raise ValueError("partial Docker byte counts")
        if self.current is not None and (type(self.current) is not int or type(self.total) is not int
                or not 0 <= self.current <= self.total or self.total <= 0):
            raise ValueError("invalid Docker byte counts")
        if self.content is not None and (not isinstance(self.content, dict)
                or self.content.get("status") not in {"complete", "incomplete", "unavailable", "unverified"}):
            raise ValueError("invalid Docker content status")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)
