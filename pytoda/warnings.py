from warnings import warn

from .types import Any


def device_warning(device: Any, throw_exception: bool = True):
    """

    Args:
        device: A device object provided by the user.
        throw_exception: Whether an exception is thrown. Defaults to True.

    Raises:
        ValueError: Raises if throw_exception is True.

    """

    if device:
        msg = (
            f"Found deprecated argument: device={device}. Device value will be ignored, "
            "all devices will be set to `cpu`. If you are using cuda devices, also "
            "remember to transfer tensors to the gpu before using a model __call__"
        )
        if throw_exception:
            raise ValueError(msg)
        warn(msg)
