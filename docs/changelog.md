# Changelog

## 0.2.2 — scale robustness (SuiteSparse dataset)

Validating all 50 solvers against 260 real matrices from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/) (n ≤ 600, spanning
structural, circuit, CFD, chemical and graph problems) exposed a systemic class of
defects: **absolute thresholds where a scale-relative test is required**. Real
matrices with very large or very small entries tripped these where synthetic,
unit-scale tests never did. All fixed in the C core:

- **Scale-relative symmetry test.** `cholesky`, `ldlt`, `eigenvalue_decomposition`
  and the symmetric Krylov solvers compared `|A[i,j] - A[j,i]|` against an absolute
  `1e-10`, rejecting genuinely-symmetric large-magnitude matrices (e.g. entries
  ~1e7) as non-SPD. The tolerance is now relative to the entry magnitudes.
- **Scale-relative singular/pivot tests.** `gaussian_elimination`, `gauss_jordan`,
  `lu_decomposition`, `triangularization`, `back_substitution`,
  `forward_substitution` and the Householder QR (`qr_decomposition`,
  `lq_decomposition`) judged a pivot/diagonal "zero" against an absolute `1e-10`,
  reporting well-conditioned tiny-entry matrices (entries ~1e-6) as singular.
  Thresholds are now scaled by the matrix magnitude. This also fixes
  `normal_equations` and `orthogonal_projection`, which route through them.
- **Partial pivoting in row-echelon methods.** `row_echelon` and
  `reduced_row_echelon` selected the first non-zero pivot instead of the
  largest-magnitude one, which is numerically unstable; they now use partial
  pivoting like `gaussian_elimination`.
- **Log-space determinants.** `cramers_rule` and `determinant` accumulated the
  determinant as a product of pivots, which under/overflows (e.g. `0.1^66`),
  reporting well-conditioned small-entry matrices as singular. They now work in
  log space (`sign` + `log|det|`) so the Cramer ratio stays stable.

Added `tests/test_scale_robustness.py` (network-free synthetic reproductions) and
an opt-in `tests/test_suitesparse.py` that validates solvers against real
downloaded matrices (`pip install -e ".[dataset]"`; skips cleanly offline).

### Known limitations (confirmed inherent, not defects)

The dataset also confirmed the textbook limits of several iterative methods on hard
real matrices — pick a different method rather than expecting these to converge:

- **Non-normal systems:** `bicgstab`, `cgs`, `tfqmr`, `bicg`, `qmr`, `cgnr`, `lsqr`
  can break down or stagnate on strongly non-normal matrices (e.g. `gre_*`, `west*`).
  `gmres` is the robust choice there.
- **Slow classical iterations:** `jacobi` and `richardson` may exhaust the iteration
  budget on large or ill-conditioned systems that `gauss_seidel`, `sor`, or
  `conjugate_gradient` solve quickly. `gradient_descent` (steepest descent) is
  similarly slow on moderately ill-conditioned SPD systems.
- **Classical Gram-Schmidt** loses orthogonality on ill-conditioned matrices; prefer
  `modified_gram_schmidt` or `givens_qr`.

## 0.2.1 — solver robustness

Adversarial stress testing of all 50 solvers surfaced a set of convergence and
pivoting defects, all now fixed in the C core:

- **Relative convergence tolerance.** `jacobi`, `gauss_seidel`, `sor`, `ssor`,
  `richardson`, `iterative_refinement`, `qmr`, `gcr`, `symmlq`, `cgnr`, and
  `bicgstab` compared the residual (or solution increment) against a fixed
  absolute threshold, so a large-magnitude right-hand side (e.g. `b * 1e6`)
  raised a spurious `ConvergenceError`. The stopping test is now scaled by the
  problem (`‖b‖` or the solution norm).
- **Zero right-hand side.** `bicgstab` and `cgnr` broke down on `b = 0` instead
  of returning the exact solution `x = 0`; both now short-circuit on a
  zero/near-zero initial residual.
- **Crout partial pivoting.** `crout` now factors `P A = L U` with partial
  pivoting, so nonsingular matrices with a zero or tiny leading pivot (and
  permutation matrices) are solved instead of reported singular.
- **Accurate spectral bounds.** `chebyshev` and `richardson` replaced loose
  Gershgorin-disk eigenvalue estimates with a cyclic-Jacobi symmetric
  eigensolver. `chebyshev` now converges on dense SPD systems; `richardson`
  uses the optimal SPD step `2 / (λ_min + λ_max)`.
- **SOR relaxation factor.** `sor` over-relaxed (`ω = 1.5`) on nonsymmetric
  systems where that diverges; it now uses `ω = 1` (Gauss-Seidel) for
  nonsymmetric matrices and reserves over-relaxation for symmetric ones.
- **BiCGSTAB breakdown recovery.** `bicgstab` now restarts with a refreshed
  shadow residual on a `ρ` breakdown rather than giving up.

### Known limitations

These are properties of the algorithms, not defects:

- **Fixed-step Richardson** converges slowly; at the `1e-10` tolerance it solves
  SPD systems up to roughly `cond ≈ 80` within the iteration budget.
- **BiCGSTAB** is not guaranteed to converge on strongly non-normal matrices;
  for those, prefer `gmres`, `bicg`, or `cgs`.
