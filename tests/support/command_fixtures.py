"""Keep command fakes faithful when callers redirect output to real files."""
import copy
import subprocess
from functools import wraps


def redirect_command_output(fake):
    @wraps(fake)
    def run(command, **kwargs):
        def redirect(result):
            for channel in ("stdout", "stderr"):
                stream = kwargs.get(channel)
                if stream is not None:
                    assert not kwargs.get("capture_output", False)
                    output = getattr(result, channel, None) or ""
                    stream.write(output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output)
                    stream.flush()
                    setattr(result, channel, None)
            return result

        try:
            result = fake(command, **kwargs)
        except subprocess.TimeoutExpired as exc:
            redirect(exc)
            raise
        return redirect(copy.copy(result))

    return run
