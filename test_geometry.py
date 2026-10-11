
import numpy as np

rng = np.random.default_rng(42)

# Minority observations exactly on the x-axis
t = np.linspace(-2.0, 2.0, 15)
minority = np.column_stack([t, np.zeros_like(t)])

n_tests = 27
failures = 0
records = []

for _ in range(n_tests):
    # Choose two distinct points
    i, j = rng.choice(len(minority), size=2, replace=False)

    lam = rng.uniform()
    z = (1.0 - lam)*minority[i] + lam*minority[j]

    x_syn, info = pca_bounded_perturbation(
        neighborhood=minority,
        z=z,
        alpha=0.05,
        eta=0.95,
        rho=1.0,
        rng=rng
    )

    line_residual = abs(x_syn[1])
    projection_residual = info["subspace_residual"]
    bound_error = max(
        0.0,
        info["displacement"] - info["bound"]
    )

    failed = (
        line_residual > 1e-10
        or projection_residual > 1e-10
        or bound_error > 1e-10
    )

    failures += int(failed)
    records.append((
        line_residual,
        projection_residual,
        bound_error
    ))

arr = np.asarray(records)

print("Generated samples:", n_tests)
print("Failed tests:", failures)
print("Maximum line residual:", arr[:, 0].max())
print("Maximum projection residual:", arr[:, 1].max())
print("Maximum displacement-bound violation:",
      arr[:, 2].max())
