"""Document checker regression coverage without external repository or network access."""

from scripts.check_docs import github_slug, validate


def test_github_slug_preserves_unicode_and_removes_punctuation():
    assert github_slug("Host command 与 diagnostics!") == "host-command-与-diagnostics"


def test_local_links_images_anchors_and_fenced_examples(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "README.md"
    target = docs / "topic.md"
    asset = docs / "figure.svg"
    target.write_text("# 中文 标题\n\n## Again\n\n## Again\n<a id='manual-id'></a>\n",
                      encoding="utf-8")
    asset.write_text("<svg/>\n", encoding="utf-8")
    source.write_text(
        "# Home\n\n[标题](topic.md#中文-标题) [duplicate](topic.md#again-1)\n"
        "[manual](topic.md#manual-id) ![Figure](figure.svg)\n"
        "<a href='topic.md#again'>html</a>\n"
        "```md\n[false link](missing.md)\n```\n", encoding="utf-8"
    )
    assert validate([source, target], tmp_path) == []


def test_missing_paths_and_anchors_are_reported(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("[missing](gone.md) [heading](#unknown)\n", encoding="utf-8")
    errors = validate([doc], tmp_path)
    assert any("broken local link" in issue for issue in errors)
    assert any("missing anchor" in issue for issue in errors)


def test_whitespace_and_missing_newline(tmp_path):
    path = tmp_path / "README.md"
    path.write_bytes(b"# Hi \nNo newline")
    errors = validate([path], tmp_path)
    assert any("trailing whitespace" in issue for issue in errors)
    assert any("missing final newline" in issue for issue in errors)


def test_skill_frontmatter_and_yaml(tmp_path):
    skill = tmp_path / ".agents/skills/example/SKILL.md"
    skill.parent.mkdir(parents=True)
    yaml_file = skill.parent / "agents/openai.yaml"
    yaml_file.parent.mkdir()
    skill.write_text("---\nname: example\ndescription: Works\n---\n# Example\n",
                     encoding="utf-8")
    yaml_file.write_text("name: [broken\n", encoding="utf-8")
    errors = validate([skill, yaml_file], tmp_path)
    assert len(errors) == 1
    assert "invalid YAML" in errors[0]
    yaml_file.write_text("name: example\n", encoding="utf-8")
    assert validate([skill, yaml_file], tmp_path) == []
    skill.write_text("---\nname: example\n---\n# Example\n", encoding="utf-8")
    assert any("invalid Skill frontmatter" in error
               for error in validate([skill], tmp_path))


def test_linked_target_outside_selected_documents(tmp_path):
    one = tmp_path / "docs/one.md"
    two = tmp_path / "extra/two.md"
    one.parent.mkdir()
    two.parent.mkdir()
    one.write_text("[outside](../extra/two.md#heading)\n", encoding="utf-8")
    two.write_text("# Heading\n", encoding="utf-8")
    assert validate([one], tmp_path) == []
