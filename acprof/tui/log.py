"""Selectable, bounded plain-text logs for the low-refresh terminal controller."""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Sequence

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.geometry import Offset
from textual.message import Message
from textual.scrollbar import ScrollDown, ScrollTo, ScrollUp
from textual.widgets import TextArea
from textual.widgets.text_area import EditResult, Selection

from acprof.tui.log_records import DETAIL, LEVEL, LogRecords, LogRow


class SelectableLog(TextArea):
    """A read-only log which keeps original text separate from visual wrapping.

    Each ``write`` appends a log entry. Reading or selecting pauses tailing until
    ``follow_tail`` is called, so incoming output cannot steal the viewport.
    Retention counts original lines, not terminal rows created by soft wrapping.
    """

    BINDINGS = [Binding("ctrl+a", "select_all", "全选日志", show=False)]
    COMPONENT_CLASSES = {"log--info", "log--success", "log--warn", "log--error", "log--debug", "log--match"}
    DEFAULT_CSS = """
    SelectableLog {
        .log--info { color: $primary; }
        .log--success { color: $success; }
        .log--warn { color: $warning; }
        .log--error { color: $error; }
        .log--debug { color: $text-muted; }
        .log--match { background: $warning; color: $background; text-style: bold; }
    }
    """

    class Updated(Message):
        """Retention, filtering or search changed (never posted while suspended)."""

    class FollowChanged(Message):
        """The user paused following, or explicitly returned to the latest output."""

        def __init__(self, log: SelectableLog, following: bool) -> None:
            super().__init__()
            self.log = log
            self.following = following

        @property
        def control(self) -> SelectableLog:
            return self.log

    def __init__(
        self,
        *,
        max_lines: int = 5000,
        max_bytes: int = 8 * 1024 * 1024,
        details: bool = True,
        wrap: bool = True,
        id: str | None = None,
        classes: str | None = None,
        name: str | None = None,
    ) -> None:
        self._mutating_log = True
        self._has_entries = False
        self._following = True
        self._max_lines = self._validate_max_lines(max_lines)
        self.records = LogRecords(self._max_lines, self._validate_max_lines(max_bytes))
        self._rows: list[LogRow] = []
        self.level_filter = "ALL"
        self.details = details
        self.search_query = ""
        self.suspended = False
        self._matches: list[tuple[int, int, int]] = []
        self._match_index = -1
        super().__init__(
            read_only=True,
            soft_wrap=wrap,
            show_cursor=False,
            show_line_numbers=False,
            highlight_cursor_line=False,
            max_checkpoints=0,
            id=id,
            classes=classes,
            name=name,
        )
        self.cursor_blink = False
        self.match_cursor_bracket = False
        self._mutating_log = False

    @staticmethod
    def _validate_max_lines(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_lines must be a positive integer")
        return value

    @property
    def max_lines(self) -> int:
        return self._max_lines

    @max_lines.setter
    def max_lines(self, value: int) -> None:
        self._max_lines = self._validate_max_lines(value)
        self.records.max_lines = self._max_lines
        self.records.trim()
        self._sync_view()

    @property
    def wrap(self) -> bool:
        return self.soft_wrap

    @wrap.setter
    def wrap(self, value: bool) -> None:
        self.soft_wrap = bool(value)
        if self.following and self.is_mounted:
            self.call_after_refresh(self._scroll_tail_if_following)

    @property
    def lines(self) -> Sequence[str]:
        """Retained original lines; visual wrapping never inserts newlines here."""
        return tuple(record.raw for record in self.records.records)

    @property
    def raw_text(self) -> str:
        return self.records.raw_text

    @property
    def discarded_lines(self) -> int:
        return self.records.discarded

    @property
    def match_count(self) -> int:
        return len(self._matches)

    def set_view(self, *, level: str | None = None, details: bool | None = None, query: str | None = None) -> None:
        if self.suspended:
            return
        if level is not None:
            if level not in {"ALL", "INFO", "WARN", "ERROR", "DEBUG"}:
                raise ValueError("unknown log level")
            self.level_filter = level
        if details is not None:
            self.details = details
        if query is not None:
            self.search_query = query
        self._match_index = -1
        self._sync_view()

    def set_suspended(self, suspended: bool) -> None:
        was_suspended = self.suspended
        self.suspended = suspended
        if was_suspended and not suspended:
            self._sync_view()

    def find_match(self, *, previous: bool = False) -> bool:
        if self.suspended or not self._matches:
            return False
        if self._match_index < 0:
            self._match_index = len(self._matches) - 1 if previous else 0
        else:
            self._match_index = (self._match_index + (-1 if previous else 1)) % len(self._matches)
        row, start, end = self._matches[self._match_index]
        self.selection = Selection((row, start), (row, end))
        self.scroll_cursor_visible(center=True, animate=False)
        return True

    def get_line(self, line_index: int) -> Text:
        line = super().get_line(line_index)
        if line_index >= len(self._rows):
            return line
        record = self._rows[line_index].record
        category = "success" if record.success else record.level.lower()
        style = self.get_component_rich_style(f"log--{category}")
        if record.level in {"WARN", "ERROR", "DEBUG"} or record.success:
            line.stylize(style)
        else:
            for match in LEVEL.finditer(line.plain[:80]):
                line.stylize(style, match.start(), match.end())
        if record.level not in {"WARN", "ERROR"}:
            for match in DETAIL.finditer(line.plain):
                line.stylize(self.get_component_rich_style("log--debug"), match.start(), match.end())
        if self.search_query:
            for match in re.finditer(re.escape(self.search_query), line.plain, re.I):
                line.stylize(self.get_component_rich_style("log--match"), match.start(), match.end())
        return line

    @property
    def following(self) -> bool:
        return self._following

    def _set_following(self, following: bool) -> None:
        if self._following != following:
            self._following = following
            self.post_message(self.FollowChanged(self, following))

    def _watch_selection(
        self, previous_selection: Selection, selection: Selection
    ) -> None:
        super()._watch_selection(previous_selection, selection)
        if not self._mutating_log and not selection.is_empty:
            self._set_following(False)

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if (
            not self._mutating_log
            and new_value < old_value
            and not self.is_vertical_scroll_end
        ):
            self._set_following(False)

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        # Keep native selection/drag behavior while pausing tail-follow.
        self._set_following(False)
        await super()._on_mouse_down(event)

    async def _on_click(self, event: events.Click) -> None:
        if event.chain not in (2, 3):
            return
        row, column = self.get_target_document_location(event)
        line = self.document[row]
        if event.chain == 3:
            self.selection = Selection((row, 0), (row, len(line)))
        else:
            for word in re.finditer(r"\w+|[^\w\s]+|\s+", line):
                if word.start() <= column < word.end():
                    self.selection = Selection((row, word.start()), (row, word.end()))
                    break
        event.stop()
        event.prevent_default()

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._set_following(False)

    def action_cursor_up(self, select: bool = False) -> None:
        # Pause on user intent, before a new entry can interrupt scrolling.
        self._set_following(False)
        self.scroll_up(animate=False, immediate=True)

    def action_cursor_page_up(self) -> None:
        self._set_following(False)
        self.scroll_to(
            y=self.scroll_y - self.scrollable_content_region.height,
            animate=False,
            immediate=True,
        )

    def action_cursor_line_start(self, select: bool = False) -> None:
        self._set_following(False)
        self.scroll_home(animate=False, immediate=True)

    def action_cursor_down(self, select: bool = False) -> None:
        self.scroll_down(animate=False, immediate=True)

    def action_cursor_page_down(self) -> None:
        self.scroll_to(
            y=self.scroll_y + self.scrollable_content_region.height,
            animate=False,
            immediate=True,
        )

    def action_cursor_line_end(self, select: bool = False) -> None:
        # End only navigates; explicitly returning to latest resumes following.
        self.scroll_end(animate=False, immediate=True)

    def _on_scroll_up(self, event: ScrollUp) -> None:
        self.action_cursor_page_up()
        event.stop()
        event.prevent_default()

    def _on_scroll_down(self, event: ScrollDown) -> None:
        self.action_cursor_page_down()
        event.stop()
        event.prevent_default()

    def _on_scroll_to(self, message: ScrollTo) -> None:
        if self._allow_scroll:
            if message.y is not None:
                self._set_following(False)
            self.scroll_to(message.x, message.y, animate=False, immediate=True)
            message.stop()
            message.prevent_default()

    def _on_resize(self) -> None:
        # The inherited resize handler rewraps first; tailing waits for layout.
        if self.following:
            self.call_after_refresh(self._scroll_tail_if_following)

    def _scroll_tail_if_following(self) -> None:
        if self.following and self.is_mounted and not self.suspended:
            self.scroll_end(animate=False, immediate=True, x_axis=False)

    def follow_tail(self) -> None:
        """Clear the selection, show the latest output, and resume following."""
        self.selection = Selection.cursor(self.document.end)
        self._set_following(True)
        self._scroll_tail_if_following()
        if self.is_mounted:
            self.call_after_refresh(self._scroll_tail_if_following)

    def _sync_view(self) -> None:
        if self.suspended:
            return
        update = self.records.project_update(level=self.level_filter, details=self.details, query=self.search_query)
        old_rows = self._rows
        rows = old_rows[:update.start] + update.rows
        selection = self.selection
        scroll_x, scroll_y = self.scroll_x, self.scroll_y
        top, _ = self.wrapped_document.offset_to_location(Offset(0, int(scroll_y)))
        within_row = scroll_y - self.wrapped_document.location_to_offset((top, 0)).y

        def relocate(location: tuple[int, int]) -> tuple[int, int]:
            row, column = location
            if not rows or row >= len(old_rows):
                return (0, 0)
            index = min(bisect_left(rows, old_rows[row].record.sequence, key=lambda item: item.record.sequence), len(rows) - 1)
            return index, min(column, len(rows[index].text))

        self._mutating_log = True
        try:
            # Retention removes the head; normal appends only edit the tail.
            # Searching/filtering may rebuild the projection, never the raw history.
            removed = (bisect_left(old_rows, rows[0].record.sequence, key=lambda item: item.record.sequence) if rows else len(old_rows)) if update.reset else 0
            if removed:
                if removed < len(old_rows):
                    self.delete((0, 0), (removed, 0), maintain_selection_offset=False)
                else:
                    super().clear()
            remaining = old_rows[removed:]
            common = update.start
            if update.reset:
                for old, new in zip(remaining, rows, strict=False):
                    if old != new:
                        break
                    common += 1
            self._rows = rows
            if common < len(remaining):
                if common == len(rows) and common:
                    start = (common - 1, len(rows[-1].text))
                else:
                    start = (common, 0)
                self.replace("\n".join(row.text for row in rows[common:]), start, self.document.end,
                             maintain_selection_offset=False)
            elif common < len(rows):
                self.insert(("\n" if remaining else "") + "\n".join(row.text for row in rows[common:]),
                            self.document.end, maintain_selection_offset=False)
            self._has_entries = bool(rows)
            self.selection = Selection(relocate(selection.start), relocate(selection.end))
            self.history.clear()
            # TextArea caches rendered strips; a query can change styles without an edit.
            self._line_cache.clear()
            if not self.following:
                anchor = relocate((top, 0))
                self.scroll_to(x=scroll_x, y=self.wrapped_document.location_to_offset(anchor).y + within_row,
                               animate=False, immediate=True, force=True)
        finally:
            self._mutating_log = False
        self._matches = ([match for match in self._matches if match[0] < update.start]
                         + [(row, match.start(), match.end()) for row, item in enumerate(rows[update.start:], update.start)
                            for match in re.finditer(re.escape(self.search_query), item.text, re.I)]) if self.search_query else []
        self.refresh()
        self._scroll_tail_if_following()
        if self.following and self.is_mounted:
            self.call_after_refresh(self._scroll_tail_if_following)
        self.post_message(self.Updated())

    def write(self, content: str) -> SelectableLog:
        """Retain original text and update the projection without stealing selection."""
        self.records.append(content)
        self._sync_view()
        return self

    def clear(self) -> EditResult:
        """Clear the log while preserving TextArea's edit-result contract."""
        self._mutating_log = True
        try:
            result = super().clear()
            self._has_entries = False
            self.records = LogRecords(self.max_lines, self.records.max_bytes)
            self._rows = []
            self._matches = []
            self._match_index = -1
        finally:
            self._mutating_log = False
        self._set_following(True)
        self.scroll_home(animate=False, immediate=True)
        self.post_message(self.Updated())
        return result

    def clear_log(self) -> SelectableLog:
        """Clear retained output and return this log for fluent writes."""
        self.clear()
        return self

    def copy_all(self) -> bool:
        """Copy the complete current view, independent of selection and scrolling."""
        text = self.text
        if not text:
            return False
        self.app.copy_to_clipboard(text)
        return True
