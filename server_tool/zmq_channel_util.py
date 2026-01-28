import os
from typing import Optional


def channel_name_to_endpoint(channel_name: str, ipc_prefix: Optional[str] = None) -> str:
    endpoint = channel_name
    if not endpoint.startswith("ipc://") and not endpoint.startswith("tcp://"):
        if os.path.splitext(os.path.basename(endpoint))[-1] != ".ipc":
            endpoint += ".ipc"
        endpoint = "ipc://" + os.path.join(ipc_prefix, endpoint)
    return endpoint


def is_ipc_endpoint(endpoint: str) -> bool:
    return endpoint.startswith("ipc://")


def ipc_to_filepath(endpoint: str) -> str:
    assert endpoint.startswith("ipc://"), "Only ipc:// endpoints are supported"
    filepath = endpoint[6:]  # Remove "ipc://" prefix
    return filepath
