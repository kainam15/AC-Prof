"""Pure review questions for incomplete model contracts."""
from __future__ import annotations


def review_questions(task_info) -> list[dict]:
    report = task_info.model_resolution.get("contract", {})
    fields = report.get("fields", {})
    pending = report.get("unresolved_fields", [])
    draft = report.get("draft_spec", {})
    inputs = draft.get("multimodal", {}).get("inputs", {})
    input_gaps = [name for name in pending if name.startswith("multimodal.inputs.")]
    questions = []
    unassigned = {name.removeprefix("pipeline.inputs."): item["value"] for name, item in fields.items()
                  if name.startswith("pipeline.inputs.") and name.removeprefix("pipeline.inputs.") not in inputs}
    targets = [name for name, item in unassigned.items() if item.get("required")]
    text_names = {"prompt", "text", "messages", "turns"}
    if "multimodal.inputs.text" in input_gaps and not text_names.intersection(targets):
        targets.extend(name for name in unassigned if name in text_names and name not in targets)
    if input_gaps:
        for target in targets:
            questions.append({"path": "multimodal.inputs." + target, "value": None,
                              "reason": "; ".join(fields[name].get("reason", name) for name in input_gaps)})
        if not targets:
            questions.append({"path": "multimodal.inputs", "value": inputs,
                              "reason": "canonical input mapping is unresolved"})
    for name in pending:
        if name in input_gaps or name == "model_spec" and questions:
            continue
        if name == "pipeline_task" and fields[name]["state"] == "ambiguous":
            config = task_info.repository_metadata.get("config.json", task_info.model_config)
            questions.append({"path": name, "value": None, "reason": fields[name]["reason"],
                              "options": sorted(config.get("custom_pipelines", {}))})
        elif name in {"dependencies", "multimodal.forward_kwargs"}:
            editable = True
            blocked = False
            if name == "dependencies":
                value = []
                editable = False
                for candidate in report.get("dependency_candidates", []):
                    if candidate.get("activation") == "inactive" or candidate.get("dependency_kind") == "main_model":
                        continue
                    choice = {"repo_id": candidate["repo_id"], "role": candidate["role"], "required": True}
                    if choice not in value:
                        value.append(choice)
                    undecided = (not candidate.get("repo_id") or candidate.get("role") == "unknown"
                                 or candidate.get("activation", "active") != "active" or "conditional" in candidate)
                    editable |= undecided
                    blocked |= not undecided and candidate.get("resolution", {}).get("status") != "pinned"
            else:
                value = draft.get("multimodal", {}).get("forward_kwargs", {})
            question = {"path": name, "value": value, "reason": fields[name].get("reason", name)}
            if name == "dependencies" and (not editable or blocked):
                # An identified dependency with a failed lookup/file selection
                # needs repair, not permission to silently omit that dependency.
                question["read_only"] = True
            questions.append(question)
        else:
            # Arbitrary source/control-flow failures cannot be fixed by accepting
            # a checkbox. They require an explicit adapter/declaration.
            questions.append({"path": name, "value": None, "reason": fields[name].get("reason", name), "read_only": True})
    return questions
