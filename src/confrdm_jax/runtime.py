"""JAX runtime setup: 64-bit precision and device selection, verified rather than assumed.

Every entry point calls :func:`configure_jax` once, before anything else touches JAX.

**Precision is not optional.** Race log-densities are precision-sensitive — the inverse
Gaussian's survival function combines two nearly-equal normal CDFs in the right tail — and
`eamax`'s reference tolerances assume float64. JAX defaults to float32 and silently
downcasts, so x64 is enabled here rather than relied on from the environment.

**A GPU that JAX cannot see is a five-hour CPU job.** The SLURM scripts request
``--gpus=1``, but if the CUDA plugin is missing or the driver is unreachable JAX falls back
to CPU without an error — the job runs, produces correct numbers, and takes two orders of
magnitude longer. ``device: gpu`` in the config therefore *requires* a GPU by default and
fails loudly when there is none, and every run logs the devices it actually got.
"""

import logging

import jax

logger = logging.getLogger(__name__)

#: Platform names `configure_jax` accepts. ``"auto"`` takes whatever JAX finds, preferring
#: an accelerator, and never fails.
PLATFORMS = ("auto", "gpu", "cpu", "tpu")

#: JAX names a GPU backend by vendor in ``jax_platforms`` but reports its devices under the
#: generic ``"gpu"``, so a CUDA device's ``.platform`` is ``"gpu"``, not ``"cuda"``. Device
#: platforms are normalized through this map before being compared with `device`.
_DEVICE_PLATFORM_ALIASES = {"cuda": "gpu", "rocm": "gpu"}


def configure_jax(device="auto", x64=True, require_device=True):
    """Set precision and platform, then report the devices JAX actually has.

    Parameters
    ----------
    device : str, optional
        One of :data:`PLATFORMS`. ``"gpu"`` maps to JAX's ``cuda`` platform.
    x64 : bool, optional
        Enable 64-bit precision. Leave on; see the module docstring.
    require_device : bool, optional
        Raise when `device` names an accelerator that is not available, instead of letting
        JAX fall back to CPU. Ignored for ``"auto"`` and ``"cpu"``, which cannot fail.

    Returns
    -------
    list of jax.Device
        The devices in use.

    Raises
    ------
    ValueError
        If `device` is not one of :data:`PLATFORMS`.
    RuntimeError
        If `require_device` is set and the requested accelerator is unavailable.
    """
    if device not in PLATFORMS:
        raise ValueError(f"device must be one of {list(PLATFORMS)}, got {device!r}.")

    if x64:
        jax.config.update("jax_enable_x64", True)

    if device != "auto":
        # `jax_platforms` is a preference list, so naming an accelerator first still leaves
        # CPU reachable for anything that cannot run on it.
        platform = "cuda" if device == "gpu" else device
        jax.config.update("jax_platforms", f"{platform},cpu" if platform != "cpu" else "cpu")

    devices = jax.devices()
    kinds = sorted({_DEVICE_PLATFORM_ALIASES.get(d.platform, d.platform) for d in devices})
    logger.info("JAX devices: %s (x64=%s)", devices, jax.config.jax_enable_x64)

    if require_device and device in ("gpu", "tpu") and device not in kinds:
        raise RuntimeError(
            f"device={device!r} was requested but JAX only sees {kinds}. A job that "
            "falls back to CPU here runs correctly and roughly two orders of magnitude "
            "slower, so this is an error rather than a warning. Install the matching "
            "JAX accelerator plugin (e.g. `pip install -U \"jax[cuda12]\"`), or set "
            "`device=auto` to accept whatever is available."
        )

    return devices
