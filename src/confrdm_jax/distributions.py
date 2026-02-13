"""Distribution utilities for hierarchical Bayesian inference.

This module provides utilities for working with the Cholesky parameterizations used in MVN hierarchical priors.

"""

import jax.numpy as jnp


def cholesky_to_flat(L):
    """Convert lower-triangular Cholesky factor to flat vector.

    Layout: [log(diag(L)), offdiag(L) row-major]
    For P=5: [log(L_00), log(L_11), ..., log(L_44), L_10, L_20, L_21, ...]

    The diagonal is stored in log-space to ensure positivity.

    Args:
        L: Lower triangular Cholesky factor (P, P).

    Returns:
        Flat vector of length P + P*(P-1)/2.
    """
    P = L.shape[0]
    log_diag = jnp.log(jnp.diag(L))

    # Extract strictly lower triangular elements (below diagonal)
    tril_indices = jnp.tril_indices(P, k=-1)
    offdiag = L[tril_indices]

    return jnp.concatenate([log_diag, offdiag])


def flat_to_cholesky(flat, P):
    """Convert flat vector to lower-triangular Cholesky factor.

    Inverse of cholesky_to_flat.

    Args:
        flat: Flat vector of length P + P*(P-1)/2.
        P: Dimension of the matrix.

    Returns:
        Lower triangular Cholesky factor (P, P).
    """
    log_diag = flat[:P]
    offdiag = flat[P:]

    L = jnp.zeros((P, P))
    L = L.at[jnp.diag_indices(P)].set(jnp.exp(log_diag))

    tril_indices = jnp.tril_indices(P, k=-1)
    L = L.at[tril_indices].set(offdiag)

    return L
