"""Shared multimodal declarations without host or inference dependencies."""


def pipeline_spec():
    return {
        "schema_version": 1,
        "format": "transformers-pipeline",
        "task": "audio-text-to-text",
        "pipeline_task": "listen-and-answer",
        "multimodal": {
            "inputs": {"waveform": "audio", "rate": "sampling_rate", "question": "text"},
            "forward_kwargs": {"limit": "$max_new_tokens", "temperature": 0.0},
        },
        "dependencies": [{
            "repo_id": "example/base", "revision": "b" * 40,
            "allow_patterns": ["*.json", "*.safetensors"],
        }],
    }
