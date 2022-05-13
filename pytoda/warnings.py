from .types import Any
from warnings import warn

def device_warning(device: Any) -> str:
    if device is None:
        msg = (
            f"Found deprecated argument: device={device}. Device value will be ignored, "
            "all devices will be set to `cpu`."
        )
    warn(msg)
    return msg
