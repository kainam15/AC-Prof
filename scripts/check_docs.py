"""Check tracked project documentation for local links, anchors and Markdown formatting."""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = MarkdownIt("commonmark")


class HtmlLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.anchors.add(values["id"])
        if tag == "a" and values.get("name"):
            self.anchors.add(values["name"])
        if tag == "a" and values.get("href"):
            self.links.append(values["href"])
        if tag in {"img", "source"} and values.get("src"):
            self.links.append(values["src"])


def github_slug(heading: str) -> str:
    text = heading.lower().strip()
    text = "".join(
        char for char in text
        if char in {"-", "_", " "} or
        (not unicodedata.category(char).startswith(("P", "S", "C")))
    )
    return re.sub(r"\s", "-", text)


def document_links(content: str) -> tuple[set[str], list[str]]:
    tokens = MARKDOWN.parse(content)
    counts: dict[str, int] = defaultdict(int)
    anchors: set[str] = set()
    links: list[str] = []
    for index, token in enumerate(tokens):
        if token.type == "heading_open" and index + 1 < len(tokens):
            inline = tokens[index + 1]
            heading = "".join(
                child.content for child in (inline.children or [])
                if child.type in {"text", "code_inline", "image"}
            )
            slug = github_slug(heading)
            number = counts[slug]
            anchors.add(f"{slug}-{number}" if number else slug)
            counts[slug] += 1
        for child in [token, *(token.children or [])]:
            if child.type == "link_open" and child.attrGet("href"):
                links.append(child.attrGet("href"))
            if child.type == "image" and child.attrGet("src"):
                links.append(child.attrGet("src"))
            if child.type in {"html_inline", "html_block"}:
                parser = HtmlLinks()
                parser.feed(child.content)
                anchors.update(parser.anchors)
                links.extend(parser.links)
    return anchors, links


def tracked_docs(root: Path) -> list[Path]:
    paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"], stderr=subprocess.DEVNULL
    ).split(b"\0")
    selected = []
    for raw in paths:
        if not raw:
            continue
        relative = Path(raw.decode("utf-8", "surrogateescape"))
        if (relative.parts[0] == "docs" and relative.suffix == ".md"
                or relative.parts[:2] == (".agents", "skills") and
                relative.suffix in {".md", ".yaml", ".yml"}
                or relative.name == "AGENTS.md"
                or str(relative) == "README.md"):
            if (root / relative).is_file():
                selected.append(root / relative)
    return selected


def validate(paths: list[Path], root: Path) -> list[str]:
    errors: list[str] = []
    metadata: dict[Path, tuple[set[str], list[str]]] = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
            content = data.decode("utf-8")
        except (UnicodeError, OSError) as exc:
            errors.append(f"{relative}: UTF-8 read failed: {exc}")
            continue
        if path.suffix in {".yaml", ".yml"}:
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as exc:
                errors.append(f"{relative}: invalid YAML: {exc}")
            continue
        if data and not data.endswith(b"\n"):
            errors.append(f"{relative}: missing final newline")
        if b"\r" in data:
            errors.append(f"{relative}: CR characters are not allowed")
        for lineno, line in enumerate(content.splitlines(), 1):
            if line[len(line.rstrip(" \t")):] not in {"", "  "}:
                errors.append(f"{relative}:{lineno}: trailing whitespace")
            if line.endswith("\t"):
                errors.append(f"{relative}:{lineno}: trailing tab")
        if relative.startswith(".agents/skills/") and path.name == "SKILL.md":
            try:
                _, front, _ = content.split("---", 2)
                fields = yaml.safe_load(front)
                if not isinstance(fields, dict) or not all(
                    isinstance(fields.get(key), str) and fields[key].strip()
                    for key in ("name", "description")
                ):
                    raise ValueError("name and description must be nonempty")
            except (ValueError, yaml.YAMLError):
                errors.append(f"{relative}: invalid Skill frontmatter")
        metadata[path] = document_links(content)

    for path, (_, links) in list(metadata.items()):
        relative = path.relative_to(root).as_posix()
        for link in links:
            try:
                parts = urlsplit(link)
            except ValueError:
                errors.append(f"{relative}: invalid link: {link}")
                continue
            if parts.scheme or parts.netloc or link.startswith("//"):
                continue
            destination = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            try:
                display = destination.relative_to(root.resolve()).as_posix()
            except ValueError:
                errors.append(f"{relative}: link escapes repository: {link}")
                continue
            if not destination.exists():
                errors.append(f"{relative}: broken local link: {link}")
            elif parts.fragment and destination.suffix == ".md" and destination.is_file():
                try:
                    anchors = metadata.get(destination)
                    if anchors is None:
                        anchors = document_links(destination.read_text(encoding="utf-8"))
                        metadata[destination] = anchors
                    if unquote(parts.fragment) not in anchors[0]:
                        errors.append(f"{relative}: missing anchor in {display}: {link}")
                except (UnicodeError, OSError) as exc:
                    errors.append(f"{relative}: link target unreadable: {exc}")
    return errors


def main() -> int:
    errors = validate(tracked_docs(ROOT), ROOT)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"Documentation check failed: {len(errors)} errors", file=sys.stderr)
        return 1
    print("Documentation links, anchors, Markdown formatting and Skill YAML: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
