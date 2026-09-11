"""Runtime configuration: precision on, and no silent CPU fallback."""

import jax
import pytest

from confrdm_jax.runtime import configure_jax


def test_x64_is_enabled():
    configure_jax("auto")
    assert jax.config.jax_enable_x64


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="device must be one of"):
        configure_jax("cuda")


def test_auto_never_fails():
    assert configure_jax("auto", require_device=True)


def test_cpu_is_always_available():
    assert configure_jax("cpu")


@pytest.mark.skipif(
    any(d.platform == "cuda" for d in jax.devices()), reason="a GPU is actually present"
)
def test_missing_gpu_is_an_error_not_a_fallback():
    # A job that requests a GPU and silently runs on CPU takes roughly two orders of
    # magnitude longer and produces correct numbers, so it is easy to miss in a log.
    with pytest.raises(RuntimeError, match="falls back to CPU"):
        configure_jax("gpu", require_device=True)
