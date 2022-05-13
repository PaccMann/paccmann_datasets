from .types import Any


def device_warning(device: Any) -> str:
    return (
        f"Found deprecated argument: device={device}. This is a legacy, support for "
        "devices was removed, all devices will be set to `cpu`."
    )
