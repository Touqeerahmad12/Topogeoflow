
import numpy as np
def boundary_safe(candidate, X_minority, X_majority,
                  gamma=0.90):
    """
    Implements manuscript Equations (16)-(19).

    Returns:
        accepted: bool
        dm: nearest minority distance
        dM: nearest majority distance
    """
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must be in (0, 1).")

    x = np.asarray(candidate, dtype=float)
    Xm = np.asarray(X_minority, dtype=float)
    XM = np.asarray(X_majority, dtype=float)

    if Xm.ndim != 2 or XM.ndim != 2:
        raise ValueError("Class supports must be 2D arrays.")
    if len(Xm) == 0 or len(XM) == 0:
        raise ValueError("Both class supports must be nonempty.")
    if Xm.shape[1] != x.size or XM.shape[1] != x.size:
        raise ValueError("Feature dimension mismatch.")

    # Eq. (16): nearest observed minority distance
    dm = float(np.min(np.linalg.norm(Xm - x, axis=1)))

    # Eq. (17): nearest observed majority distance
    dM = float(np.min(np.linalg.norm(XM - x, axis=1)))

    # Eq. (19): exact acceptance rule
    accepted = (
        np.isfinite(dm)
        and np.isfinite(dM)
        and dM > 0.0
        and dm <= gamma * dM
    )

    return bool(accepted), dm, dM

def pca_bounded_perturbation(
    neighborhood,
    z,
    alpha=0.05,
    eta=0.95,
    rho=1.0,
    rng=None
):
    """
    Implements manuscript Eqs. (5)-(13).

    neighborhood: (n_local, d) minority observations
    z: (d,) interpolation point
    Returns: synthetic point and diagnostic information
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(neighborhood, dtype=float)
    z = np.asarray(z, dtype=float)

    if X.ndim != 2 or X.shape[1] != z.size:
        raise ValueError("Incompatible neighborhood and z.")
    if X.shape[0] < 2 or not (0 < eta < 1):
        raise ValueError("Invalid neighborhood or PCA threshold.")
    if alpha <= 0 or rho <= 0:
        raise ValueError("alpha and rho must be positive.")

    # Eq. (5): local mean
    mu = X.mean(axis=0)
    centered = X - mu
    n, d = X.shape

    # Eq. (6): local covariance, normalized by n
    C = centered.T @ centered / n

    # Eq. (7): eigendecomposition and intrinsic rank
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    total = eigenvalues.sum()
    if total <= 1e-14:
        raise ValueError("Degenerate local neighborhood.")

    cumulative = np.cumsum(eigenvalues) / total
    q = int(np.searchsorted(cumulative, eta) + 1)

    # Eq. (8): principal-subspace projector
    Uq = eigenvectors[:, :q]
    P = Uq @ Uq.T

    # Eq. (9): local projected scale
    projected = centered @ Uq
    sigma_sq = np.sum(projected**2) / n
    sigma = np.sqrt(sigma_sq)

    # Eqs. (10)-(11): Gaussian noise and PCA projection
    gaussian_std = alpha * sigma / np.sqrt(q)
    xi = rng.normal(0.0, gaussian_std, size=d)
    epsilon_raw = Uq @ (Uq.T @ xi)

    # Eq. (12): radial truncation
    bound = rho * sigma
    raw_norm = np.linalg.norm(epsilon_raw)
    if raw_norm > bound and raw_norm > 0:
        epsilon = epsilon_raw * (bound / raw_norm)
    else:
        epsilon = epsilon_raw

    # Eq. (13): final synthetic point
    x_syn = z + epsilon

    # Numerical verification of Proposition 2
    subspace_residual = np.linalg.norm(epsilon - P @ epsilon)
    displacement = np.linalg.norm(epsilon)

    tol = 1e-10 * max(1.0, sigma)
    assert subspace_residual <= tol
    assert displacement <= bound + tol

    diagnostics = {
        "q": q,
        "sigma": sigma,
        "bound": bound,
        "displacement": displacement,
        "subspace_residual": subspace_residual,
        "interpolation_point": z,
        "perturbation": epsilon,
        "projector": P
    }

    return x_syn, diagnostics
