"""Check tracked project documentation for local links, anchors and Markdown formatting."""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = MarkdownIt("commonmark").enable(["table", "strikethrough"])

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



# CommonMark can leave emphasis markers literal when punctuation is followed by CJK text.
_CJK_EMPHASIS = re.compile(
    r"(?<![*_\\])(?P<mark>\*\*|__|\*|_)(?P<body>[^*_\n]{1,100}?)(?P=mark)(?=[\u3400-\u9fff])"
)


def invalid_cjk_emphasis(content: str) -> list[tuple[int, str]]:
    """Only inspect rendered prose: ignore code spans, fences, and escaped markers."""
    lines = content.splitlines()
    problems: set[tuple[int, str]] = set()
    for token in MARKDOWN.parse(content):
        if token.type != "inline" or token.map is None:
            continue
        start, end = token.map
        for child in token.children or []:
            if child.type != "text":
                continue
            for match in _CJK_EMPHASIS.finditer(child.content):
                snippet = match.group(0)
                for index in range(start, min(end, len(lines))):
                    source = lines[index]
                    pos = source.find(snippet)
                    if pos >= 0 and (pos == 0 or source[pos - 1] != "\\"):
                        problems.add((index + 1, snippet))
                        break
    return sorted(problems)


_README_SECTION_LABELS = {
    "功能概览": "Capabilities",
    "快速开始": "Quick start",
    "采集与分析": "Measurement and analysis",
    "运行环境": "Platform support",
    "文档与开发": "Documentation and development",
    "贡献与安全": "Contributing and security",
    "许可与来源": "License and origins",
}


def readme_signature(path: Path, root: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]], Counter[str], Counter[str]]:
    """Language-neutral document structure; does not claim semantic translation."""
    content = path.read_text(encoding="utf-8")
    tokens = MARKDOWN.parse(content)
    headings = []
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            heading = "".join(child.content for child in (tokens[index + 1].children or [])
                              if child.type in {"text", "code_inline"})
            headings.append((token.tag, _README_SECTION_LABELS.get(heading, heading)))
    fences = [(token.info.strip(), token.content) for token in tokens if token.type == "fence"]
    spans = Counter(child.content for token in tokens for child in (token.children or [])
                    if child.type == "code_inline")
    _, links = document_links(content)
    targets: Counter[str] = Counter()
    for link in links:
        parts = urlsplit(link)
        if parts.scheme or parts.netloc or link.startswith("//"):
            target = link
        else:
            resolved = (path.parent / unquote(parts.path)).resolve()
            try:
                relative = resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                continue  # Already covered by link path validation.
            if relative in {"README.md", "docs/i18n/README_zh-CN.md"}:
                continue  # Reciprocal language links point to different READMEs.
            target = f"{relative}?{parts.query}#{unquote(parts.fragment)}"
        targets[target] += 1
    return headings, fences, spans, targets


def check_readme_parity(root: Path) -> list[str]:
    """Detect divergent heading structure, command examples, and link targets."""
    english = root / "README.md"
    chinese = root / "docs/i18n/README_zh-CN.md"
    if not english.is_file() or not chinese.is_file():
        return []
    left, right = readme_signature(english, root), readme_signature(chinese, root)
    labels = ("heading structure", "fenced examples", "inline code", "link targets")
    return [f"README bilingual parity: {name} differs between README.md and "
            "docs/i18n/README_zh-CN.md"
            for name, a, b in zip(labels, left, right, strict=True) if a != b]


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
        for lineno, snippet in invalid_cjk_emphasis(content):
            errors.append(
                f"{relative}:{lineno}: emphasis is rendered literally before CJK text: {snippet}"
            )
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
    english = root / "README.md"
    chinese = root / "docs/i18n/README_zh-CN.md"
    if english in metadata and chinese in metadata:
        errors.extend(check_readme_parity(root))
    return errors


def main() -> int:
    errors = validate(tracked_docs(ROOT), ROOT)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"Documentation check failed: {len(errors)} errors", file=sys.stderr)
        return 1
    print("Documentation links, GFM text rendering, README parity and Skill YAML: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
