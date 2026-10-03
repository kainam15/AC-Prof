"""Permission installation runs only a reviewed, fixed system-tool plan."""
import json
import subprocess
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.host import permissions


class TestPermissionSetup:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        directory = tmp_path
        self.root = Path(str(directory))
        for name in ('sudo', 'perf', 'tcpdump', 'groupadd', 'chown', 'chmod', 'setfacl', 'setcap', 'getfacl', 'getcap'):
            (self.root / name).write_bytes(b'\x7fELFtest-only')
        for context in (patch.object(permissions, 'system_executable', side_effect=lambda tool_name: self.root / tool_name),
                        patch.object(permissions.os, 'getuid', return_value=1000),
                        patch.object(permissions.platform, 'system', return_value='Linux')):
            context.start()
            self._request.addfinalizer(partial(context.stop))

    def test_review_does_not_execute_and_grants_only_selected_capability(self):
        with patch.object(permissions, 'run_command') as run:
            plan = permissions.build_permission_plan(('perf',))
        run.assert_not_called()
        assert ([target.name for target in plan.targets]) == (['perf'])
        assert ((str(self.root / 'setcap'), 'cap_perfmon=ep', str(self.root / 'perf'))) in (plan.commands)
        assert ((str(self.root / 'setfacl'), '-m', 'u:1000:r-x', str(self.root / 'perf'))) in (plan.commands)
        assert not (any('sysctl' in str(command) or 'cap_sys_admin' in str(command) for command in plan.commands))

    def test_modified_plan_cannot_run_arbitrary_root_commands(self):
        plan = permissions.build_permission_plan(('perf',))
        forged = replace(plan, commands=((str(self.root / 'sudo'), 'sh', '-c', 'unreviewed'),))
        with patch.object(permissions, 'run_command') as run:
            with pytest.raises(ValueError, match='review'):
                permissions.execute_permission_plan(forged, backup_path=self.root / 'backup.json')
        run.assert_not_called()

    def test_failure_stops_install_and_preserves_original_permission_snapshot(self):
        plan = permissions.build_permission_plan(('perf',))
        backup = self.root / 'backup.json'

        def run(command, **_kwargs):
            if Path(command[0]).name in {'getfacl', 'getcap'}:
                return subprocess.CompletedProcess(command, 0, stdout='original permission state\n')
            assert (backup.exists()), 'Save original permissions before the first mutation'
            return subprocess.CompletedProcess(command, 1)

        with patch.object(permissions, 'run_command', side_effect=run) as runner:
            with pytest.raises(RuntimeError, match='step 1/'):
                permissions.execute_permission_plan(plan, backup_path=backup)
        assert (runner.call_count) == (3)
        assert (json.loads(backup.read_text())['targets'][0]['getfacl']) == ('original permission state\n')
        assert ('input') not in (runner.call_args.kwargs)
        assert ('shell') not in (runner.call_args.kwargs)

    def test_executable_replaced_after_review_requires_new_plan(self):
        plan = permissions.build_permission_plan(('perf',))
        (self.root / 'perf').write_bytes(b'\x7fELFchanged-after-review')
        with patch.object(permissions, 'run_command') as run:
            with pytest.raises(ValueError):
                permissions.execute_permission_plan(plan, backup_path=self.root / 'backup.json')
        run.assert_not_called()

    def test_user_writable_executable_is_never_eligible_for_capabilities(self):
        path = self.root / 'perf'
        path.chmod(0o777)
        with pytest.raises(ValueError, match='writable executable'):
            permissions._trusted_file(path, elf=True)
