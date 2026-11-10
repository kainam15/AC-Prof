"""Pure Docker port policy shared by configuration and service launch."""
from acprof.config import SERVER_PORT


def service_port(cpu: int, mem: int) -> int:
    """Keep the existing mapping; sequential cases may reuse a host port."""
    port = SERVER_PORT + cpu * 100 + mem
    if not 1 <= port <= 65535:
        raise ValueError(
            f"Docker service port {port} for cpus={cpu}, mems={mem} is outside 1..65535; "
            "choose a smaller Docker resource combination"
        )
    return port
