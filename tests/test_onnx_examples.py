"""Prepared snapshot provenance must still describe the files actually validated."""
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from examples.onnxruntime import real_models


def test_modified_tokenizer_is_rejected_before_runtime_import():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'tokenizer.json').write_bytes(b'modified')
        declaration = {'repository': 'local/test', 'revision': 'test', 'license': {}, 'family': 'nlp',
                       'files': {'tokenizer.json': hashlib.sha256(b'original').hexdigest()}}
        with patch.dict(real_models.MODELS, {'test': declaration}), patch(
                'acprof.container.handlers.HandlerRegistry.get',
                side_effect=AssertionError('runtime selected before checking modified tokenizer')):
            with pytest.raises(ValueError, match='tokenizer.json.*hash'):
                real_models.validate('test', root)

def test_report_does_not_trust_a_modified_provenance_sidecar():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'model.onnx').write_bytes(b'original')
        (root / 'source_provenance.json').write_text(json.dumps({'revision': 'invented'}))
        declaration = {'repository': 'local/test', 'revision': 'test', 'license': {}, 'family': 'nlp',
                       'files': {'model.onnx': hashlib.sha256(b'original').hexdigest()}}
        with patch.dict(real_models.MODELS, {'test': declaration}), patch(
                'acprof.container.handlers.HandlerRegistry.get',
                side_effect=AssertionError('runtime selected before checking modified provenance')):
            with pytest.raises(ValueError, match='provenance'):
                real_models.validate('test', root)
