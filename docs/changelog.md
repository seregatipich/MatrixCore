# Changelog

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
