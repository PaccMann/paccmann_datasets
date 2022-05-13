from .types import Any
from warnings import warn

def device_warning(device: Any, throw_exception=True) -> str:
    if device is None:
        msg = (
            f"Found deprecated argument: device={device}. Device value will be ignored, "
            "all devices will be set to `cpu`."
        )
    if throw_exception:
        raise msg
    warn(msg)
    return msg
