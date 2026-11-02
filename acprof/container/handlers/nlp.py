"""NLP task handler - text-generation, classification, fill-mask, QA, etc."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from acprof.container.handlers import (
    BaseHandler,
    handler_declaration,
    model_revision_kwargs,
    transformers_pipeline_load_kwargs,
)
from acprof.container.handlers.generation_evidence import count_input_tokens, generation_evidence
from acprof.container.handlers.text_generation import text_output_metrics
from acprof.container.handlers.text_inputs import (
    get_model_max_length,
    max_effective_qa_context_length,
    max_effective_single_text_length,
    string_list,
    truncate_qa_context,
    truncate_single_text,
)
from acprof.container.load_policy import load_policy
from acprof.input_budget import check_nlp_input, positive_integer
from acprof.models.spec import pipeline_task

# Tasks that generate text output
_GENERATIVE_TASKS = {
    "text-generation", "text2text-generation", "summarization",
    "translation", "conversational",
}


class NLPHandler(BaseHandler):

    @staticmethod
    def _generation_token_reserve(pipe: Any, task_type: str, params: dict[str, Any]) -> int:
        config = getattr(getattr(pipe, "model", None), "config", None)
        if task_type != "text-generation" or getattr(config, "is_encoder_decoder", False) is True:
            return 0
        value = params.get("max_new_tokens", 64)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        return value

    def load(
        self,
        model_source: str,
        task_type: str,
        backend: str,
        device: str,
        model_revision: str = "main",
        load_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        encoder = handler_declaration(task_type, backend).handler_options.get("encoder", "")
        if encoder:
            from acprof.container.handlers.sentence_encoders import SentenceEncoderHandler
            delegate = SentenceEncoderHandler(encoder)
            context = delegate.load(model_source, task_type, backend, device, model_revision,
                                    load_options=load_options)
            context["_encoder_handler"] = delegate
            return context

        device_map = device if device == "cpu" else "auto"
        policy = load_policy(model_source, task_type, backend, device, load_options)
        torch_dtype = policy["dtype"]
        pipeline_options = transformers_pipeline_load_kwargs(load_options)
        if backend == "transformers_model" and task_type in {"text2text-generation", "summarization", "translation"}:
            from acprof.container.handlers.text_generation import load_seq2seq
            pipe = load_seq2seq(model_source, model_revision, device=device, dtype=torch_dtype,
                                model_kwargs=pipeline_options.get("model_kwargs", {}))
        elif task_type == "replaced-token-detection":
            if backend != "transformers_model":
                raise ValueError("replaced-token-detection requires transformers_model")
            # No Transformers pipeline implements the ELECTRA discriminator's
            # binary per-token head. Resolve only a vetted architecture, never
            # execute arbitrary Hub code or infer from model_type alone.
            from transformers import AutoConfig, AutoTokenizer, ElectraForPreTraining

            revision_args = model_revision_kwargs(model_source, model_revision)
            local_args = {"local_files_only": True} if os.path.isdir(model_source) else {}
            config = AutoConfig.from_pretrained(model_source, trust_remote_code=False,
                                                **revision_args, **local_args)
            if (config.model_type != "electra"
                    or list(getattr(config, "architectures", ()) or ()) != ["ElectraForPreTraining"]):
                raise ValueError("replaced-token-detection requires an ElectraForPreTraining checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=False,
                                                      **revision_args, **local_args)
            model_kwargs = pipeline_options.get("model_kwargs", {})
            model = ElectraForPreTraining.from_pretrained(
                model_source, dtype=torch_dtype, trust_remote_code=False,
                **revision_args, **local_args, **model_kwargs).to(device).eval()
            pipe = SimpleNamespace(model=model, tokenizer=tokenizer)
        else:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                task=pipeline_task(model_source, task_type),
                model=model_source,
                **model_revision_kwargs(model_source, model_revision),
                **pipeline_options,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=policy["trust_remote_code"],
            )
            if task_type in _GENERATIVE_TASKS:
                from transformers import set_seed
                set_seed(0)
            if task_type == "text-generation":
                tokenizer = getattr(pipe, "tokenizer", None)
                if tokenizer is not None:
                    tokenizer.padding_side = "left"
                    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                        pipe.generation_config.pad_token_id = tokenizer.eos_token_id
        context = {
            "pipeline": pipe,
            "task_type": task_type,
            "backend": backend,
            "device": device,
            "model_revision": model_revision or "main",
            "load_options": dict(load_options or {}),
        }
        self._observe_generation(context)
        return context

    @staticmethod
    def _observe_generation(model_ctx: dict[str, Any]) -> None:
        """Observe existing CPU pipeline values; serial scenario, no extra inference/tokenization."""
        pipe = model_ctx["pipeline"]
        if model_ctx["task_type"] not in _GENERATIVE_TASKS or not all(
            callable(getattr(pipe, name, None)) for name in ("preprocess", "postprocess")
        ):
            return
        state = model_ctx["_generation_observation"] = {"inputs": [], "outputs": []}
        preprocess, postprocess = pipe.preprocess, pipe.postprocess

        def observe_input(*args, **kwargs):
            prepared = preprocess(*args, **kwargs)
            if isinstance(prepared, Mapping):
                state["inputs"].append({key: prepared.get(key) for key in ("input_ids", "attention_mask")})
            return prepared

        def observe_output(outputs, *args, **kwargs):
            if isinstance(outputs, Mapping):
                state["outputs"].append({key: outputs.get(key) for key in ("generated_sequence", "output_ids", "input_ids")})
            return postprocess(outputs, *args, **kwargs)

        pipe.preprocess, pipe.postprocess = observe_input, observe_output

    @staticmethod
    def _observed_generation(model_ctx: dict[str, Any]) -> dict[str, Any]:
        state = model_ctx.get("_generation_observation", {})
        pipe = model_ctx["pipeline"]
        config = copy.deepcopy(getattr(pipe, "generation_config", getattr(getattr(pipe, "model", None), "generation_config", None)))
        params = state.get("params", {})
        if config is not None:
            for key, value in params.items():
                setattr(config, key, value)
        encoder_decoder = bool(getattr(getattr(getattr(pipe, "model", None), "config", None), "is_encoder_decoder", False))
        inputs = [count_input_tokens(value) for value in state.get("inputs", [])]
        counts: list[int] = []
        reasons: list[str] = []
        complete = len(state.get("outputs", [])) == state.get("batch_size")
        for output in state.get("outputs", []):
            prompt = output.get("input_ids")
            if not encoder_decoder and prompt is None:
                complete = False
                continue
            prompt_length = int(prompt.shape[-1]) if getattr(prompt, "shape", None) is not None else 0
            generated = output.get("output_ids") if encoder_decoder else output.get("generated_sequence")
            budget = params.get("max_new_tokens", state.get("budget"))
            if budget is None and "max_length" in params:
                budget = params["max_length"] - (1 if encoder_decoder else prompt_length)
            evidence = generation_evidence(generated, config, prompt_length=prompt_length,
                                           encoder_decoder=encoder_decoder, max_new_tokens=budget)
            if evidence["actual_output_tokens_per_sequence"] is None:
                complete = False
            counts.extend(evidence["actual_output_tokens_per_sequence"] or [])
            reasons.extend(evidence["stop_reason_per_sequence"] or [])
        if not complete:
            counts, reasons = [], []
        return {"generation_parameters": params, "inference_interface": type(pipe).__name__,
                "actual_input_tokens": sum(inputs) if inputs and None not in inputs and len(inputs) == state.get("batch_size") else None,
                "actual_output_tokens": sum(counts) if counts else None,
                "actual_output_tokens_per_sequence": counts or None,
                "stop_reason": reasons[0] if reasons and len(set(reasons)) == 1 else None,
                "stop_reason_per_sequence": reasons or None}

    @staticmethod
    def _batch_size(raw_input: dict[str, Any]) -> int:
        return positive_integer(raw_input.get("batch_size", 1), "batch_size")

    def _zero_shot_hypotheses(self, pipe: Any, raw_input: dict[str, Any]) -> tuple[list[str], str, str]:
        labels = string_list(raw_input, "candidate_labels")
        template = raw_input.get("hypothesis_template", "This example is {}.")
        if not isinstance(template, str) or "{}" not in template:
            raise ValueError("hypothesis_template must contain a {} placeholder")
        hypotheses = [template.format(label) for label in labels]
        tokenizer = getattr(pipe, "tokenizer", None)
        longest = max(hypotheses, key=lambda text: len(
            tokenizer.encode(text, add_special_tokens=False)
        ) if tokenizer else len(text))
        return labels, template, longest

    def _preprocess_table(self, pipe: Any, raw_input: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        table = raw_input.get("table")
        if not isinstance(table, dict) or not table or not all(
            isinstance(name, str) and isinstance(column, list)
            for name, column in table.items()
        ):
            raise ValueError("table must be a nonempty mapping of column names to lists")
        row_counts = {len(column) for column in table.values()}
        if len(row_counts) != 1 or 0 in row_counts:
            raise ValueError("table columns must contain the same number of rows, greater than zero")
        if not all(isinstance(value, str) for column in table.values() for value in column):
            raise ValueError("table cells must be strings for the table QA tokenizer")
        query = raw_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("table-question-answering requires a nonempty query")
        frame = pd.DataFrame(table)
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("table-question-answering requires a table tokenizer")
        encoded = tokenizer(frame, query, truncation=False)
        token_ids = encoded["input_ids"]
        if len(token_ids) and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        max_len = get_model_max_length(pipe)
        if max_len is not None and len(token_ids) > max_len:
            raise ValueError(f"table exceeds model token limit ({len(token_ids)} > {max_len}); reduce table_rows")
        return {"table": frame, "query": query, "_effective_input_scale": next(iter(row_counts)),
                "_truncated_by_limit": False, "_probe_reason": "complete_table_rows_no_truncation"}

    def preprocess(self, model_ctx: dict[str, Any], raw_input: dict[str, Any]) -> Any:
        delegate = model_ctx.get("_encoder_handler")
        if delegate is not None:
            return delegate.preprocess(model_ctx, raw_input)
        task_type = model_ctx["task_type"]
        text = raw_input.get("text", "")
        params = raw_input.get("params", {})
        pipe = model_ctx["pipeline"]
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        budget = check_nlp_input(raw_input, task_type)
        common = {"params": params, "batch_size": self._batch_size(raw_input),
                  "input_budget": budget.as_dict()}
        if (task_type == "feature-extraction"
                and set(params) & {"prompt", "prompt_name", "normalize_embeddings"}):
            raise ValueError("sentence embedding parameters require backend=sentence_transformers")

        if task_type == "table-question-answering":
            return {**common, **self._preprocess_table(pipe, raw_input)}

        if task_type == "zero-shot-classification":
            labels, template, longest = self._zero_shot_hypotheses(pipe, raw_input)
            text, scale, truncated, reason = truncate_qa_context(pipe, longest, text)
            if scale == 0:
                raise ValueError("candidate hypotheses leave no model token budget for text")
            return {**common, "text": text, "candidate_labels": labels,
                    "hypothesis_template": template, "_effective_input_scale": scale,
                    "_truncated_by_limit": truncated, "_probe_reason": reason}

        if task_type == "question-answering":
            question = raw_input.get("question", text)
            context = raw_input.get("context", text)
            context, effective_input_scale, truncated_by_limit, reason = truncate_qa_context(
                pipe,
                question,
                context,
            )
            if effective_input_scale == 0 or not context.strip():
                raise ValueError("question-answering requires context tokens within the model token budget")
            return {
                **common,
                "question": question,
                "context": context,
                "params": params,
                "_effective_input_scale": effective_input_scale,
                "_truncated_by_limit": truncated_by_limit,
                "_probe_reason": reason,
            }

        text, effective_input_scale, truncated_by_limit, reason = truncate_single_text(
            pipe,
            text,
            task_type,
            reserved_tokens=self._generation_token_reserve(pipe, task_type, params),
        )
        return {
            **common,
            "text": text,
            "params": params,
            "_effective_input_scale": effective_input_scale,
            "_truncated_by_limit": truncated_by_limit,
            "_probe_reason": reason,
        }

    def get_scale_metadata(self, model_ctx: dict[str, Any], raw_input: dict[str, Any]) -> dict[str, Any]:
        delegate = model_ctx.get("_encoder_handler")
        if delegate is not None:
            return delegate.get_scale_metadata(model_ctx, raw_input)
        task_type = model_ctx["task_type"]
        pipe = model_ctx["pipeline"]

        if task_type == "table-question-answering":
            return {"input_scale_type": "table_rows", "max_effective_input_scale": None,
                    "reason": "table capacity depends on tokenized cells and query; no row truncation"}

        if task_type == "zero-shot-classification":
            fixed_text = self._zero_shot_hypotheses(pipe, raw_input)[2]
            available, reason = max_effective_qa_context_length(pipe, fixed_text)
            return {"input_scale_type": "seq_length", "max_effective_input_scale": available,
                    "reason": reason}

        if task_type == "question-answering":
            question = raw_input.get("question", raw_input.get("text", ""))
            available, reason = max_effective_qa_context_length(pipe, question)
            return {
                "input_scale_type": "seq_length",
                "max_effective_input_scale": available,
                "reason": reason,
            }

        reserve = self._generation_token_reserve(pipe, task_type, raw_input.get("params", {}))
        available, reason = max_effective_single_text_length(pipe, reserved_tokens=reserve)
        return {
            "input_scale_type": "seq_length",
            "max_effective_input_scale": available,
            "reason": reason,
        }

    # Pipelines whose _sanitize_parameters does not accept truncation directly.
    # For these, pass truncation via tokenizer_kwargs instead.
    _TRUNCATION_VIA_KWARGS_TASKS = {"fill-mask"}

    def predict(self, model_ctx: dict[str, Any], processed_input: Any) -> Any:
        delegate = model_ctx.get("_encoder_handler")
        if delegate is not None:
            return delegate.predict(model_ctx, processed_input)
        pipe = model_ctx["pipeline"]
        task_type = model_ctx["task_type"]
        params = processed_input.get("params", {})
        batch_size = processed_input.get("batch_size", 1)
        check_nlp_input(processed_input, task_type)

        if model_ctx.get("backend") == "transformers_model" and task_type in {"text2text-generation", "summarization", "translation"}:
            from acprof.container.handlers.text_generation import generate
            return generate(model_ctx, processed_input)

        if task_type == "replaced-token-detection":
            import torch
            tokenizer = pipe.tokenizer
            encoded = tokenizer([processed_input["text"]] * batch_size, return_tensors="pt",
                                padding=True, truncation=False)
            encoded = {name: tensor.to(pipe.model.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                return pipe.model(**encoded).logits

        if task_type == "table-question-answering":
            # TAPAS batches queries natively for one table. The generic
            # Pipeline batch iterator cannot unpack its nested model_inputs
            # output, so keep the outer iterator at one table per item.
            query = processed_input["query"]
            return pipe(table=processed_input["table"],
                        query=[query] * batch_size if batch_size > 1 else query,
                        batch_size=1, truncation=False, sequential=False)

        if task_type == "question-answering":
            if batch_size > 1:
                qa_sample = {"question": processed_input["question"], "context": processed_input["context"]}
                return pipe([dict(qa_sample) for _ in range(batch_size)], batch_size=batch_size)
            return pipe(
                question=processed_input["question"],
                context=processed_input["context"],
            )
        text = processed_input["text"]
        request_inputs = [text] * batch_size if batch_size > 1 else text
        batch_kwargs = {"batch_size": batch_size} if batch_size > 1 else {}
        if task_type == "zero-shot-classification":
            return pipe(request_inputs, candidate_labels=processed_input["candidate_labels"],
                        hypothesis_template=processed_input["hypothesis_template"],
                        multi_label=bool(params.get("multi_label", False)), **batch_kwargs)
        if task_type in _GENERATIVE_TASKS:
            generation_params = dict(params)
            if not generation_params.keys() & {"max_length", "max_new_tokens"}:
                generation_params["max_new_tokens"] = 64
            generation_params.setdefault("do_sample", False)
            state = model_ctx.get("_generation_observation")
            if state is not None:
                state.update(inputs=[], outputs=[], params=generation_params, batch_size=batch_size)
            return pipe(
                request_inputs,
                **generation_params,
                truncation=True,
                **batch_kwargs,
            )
        elif task_type in self._TRUNCATION_VIA_KWARGS_TASKS:
            return pipe(request_inputs, tokenizer_kwargs={"truncation": True}, **batch_kwargs)
        elif task_type == "token-classification":
            # Transformers 4.x token classification truncates internally and
            # its sanitizer rejects a caller-supplied truncation keyword.
            return pipe(request_inputs, **batch_kwargs)
        else:
            return pipe(request_inputs, truncation=True, **batch_kwargs)

    def postprocess(self, model_ctx: dict[str, Any], raw_output: Any) -> dict[str, Any]:
        delegate = model_ctx.get("_encoder_handler")
        if delegate is not None:
            return delegate.postprocess(model_ctx, raw_output)
        task_type = model_ctx["task_type"]

        if task_type == "replaced-token-detection":
            shape = list(getattr(raw_output, "shape", ()))
            if len(shape) != 2 or any(dimension <= 0 for dimension in shape):
                raise ValueError("replaced-token-detection requires [batch, sequence] logits")
            return {"task": task_type, "output_type": "token_scores", "output_shape": shape,
                    "n_results": shape[0]}

        if isinstance(raw_output, list):
            n_results = len(raw_output)
        elif isinstance(raw_output, dict):
            n_results = 1
        else:
            n_results = 1

        return {
            "task": task_type,
            "output_type": ("text" if task_type in _GENERATIVE_TASKS
                            else {"table-question-answering": "table_answer",
                                  "feature-extraction": "embedding"}.get(task_type, "label")),
            "n_results": n_results,
            **(text_output_metrics(getattr(model_ctx["pipeline"], "tokenizer", None), raw_output)
               if task_type in _GENERATIVE_TASKS else {}),
            **(model_ctx.get("generation_evidence") or self._observed_generation(model_ctx) if task_type in _GENERATIVE_TASKS else {}),
        }
