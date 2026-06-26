/**
 * MatrixCore - C implementation of linear system solvers
 *
 * This file contains implementations of all the solvers mentioned
 * in the library documentation for solving linear systems Ax = b
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "solvers.h"

#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-10

/* Populate an optional solver_info, ignoring a NULL pointer. */
static void set_info(solver_info *info, int iterations, double residual, int error_code) {
    if (info) {
        info->iterations = iterations;
        info->residual = residual;
        info->error_code = error_code;
    }
}

/* 2-norm of the residual ||A x - b|| for an n x n system, using the original A and b. */
static double residual_norm(const double *A, const double *b, const double *x, int n) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double row = -b[i];
        for (int j = 0; j < n; j++) {
            row += A[i * n + j] * x[j];
        }
        sum_sq += row * row;
    }
    return sqrt(sum_sq);
}

/* Determinant of an n x n matrix via Gaussian elimination with partial pivoting.
   Works on an internal copy, leaving the caller's matrix untouched. Returns 0 on
   allocation failure (caller treats a zero determinant as singular). */
static double matrix_determinant(const double *M, int n) {
    double *work = (double*)malloc(n * n * sizeof(double));
    if (!work) {
        return 0.0;
    }
    memcpy(work, M, n * n * sizeof(double));

    double det = 1.0;
    for (int k = 0; k < n; k++) {
        int pivot_row = k;
        double pivot_value = fabs(work[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double candidate = fabs(work[i * n + k]);
            if (candidate > pivot_value) {
                pivot_value = candidate;
                pivot_row = i;
            }
        }

        if (pivot_value < TOLERANCE) {
            free(work);
            return 0.0;
        }

        if (pivot_row != k) {
            for (int j = 0; j < n; j++) {
                double temp = work[k * n + j];
                work[k * n + j] = work[pivot_row * n + j];
                work[pivot_row * n + j] = temp;
            }
            det = -det;
        }

        det *= work[k * n + k];
        for (int i = k + 1; i < n; i++) {
            double factor = work[i * n + k] / work[k * n + k];
            for (int j = k + 1; j < n; j++) {
                work[i * n + j] -= factor * work[k * n + j];
            }
        }
    }

    free(work);
    return det;
}

/* One-sided Jacobi SVD of a square matrix A (n x n).
   On success U_scaled holds the (unnormalized) left factors whose columns have
   norm sigma_j, V holds the right singular vectors as columns, and sigma the
   singular values. Returns SOLVER_SUCCESS or SOLVER_NOT_CONVERGED. */
static int one_sided_jacobi_svd(const double *A, int n, double *U_scaled,
                                double *V, double *sigma, int *sweeps_out) {
    memcpy(U_scaled, A, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    const int max_sweeps = 100;
    int sweep;
    int converged = 0;
    for (sweep = 0; sweep < max_sweeps; sweep++) {
        double max_offdiag = 0.0;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int k = 0; k < n; k++) {
                    double up = U_scaled[k * n + p];
                    double uq = U_scaled[k * n + q];
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }

                double denom = sqrt(alpha * beta);
                if (denom == 0.0) {
                    continue;
                }
                double ratio = fabs(gamma) / denom;
                if (ratio > max_offdiag) {
                    max_offdiag = ratio;
                }
                if (ratio <= TOLERANCE) {
                    continue;
                }

                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = (zeta >= 0.0)
                    ? 1.0 / (zeta + sqrt(1.0 + zeta * zeta))
                    : -1.0 / (-zeta + sqrt(1.0 + zeta * zeta));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = c * t;

                for (int k = 0; k < n; k++) {
                    double up = U_scaled[k * n + p];
                    double uq = U_scaled[k * n + q];
                    U_scaled[k * n + p] = c * up - s * uq;
                    U_scaled[k * n + q] = s * up + c * uq;
                    double vp = V[k * n + p];
                    double vq = V[k * n + q];
                    V[k * n + p] = c * vp - s * vq;
                    V[k * n + q] = s * vp + c * vq;
                }
            }
        }

        if (max_offdiag < TOLERANCE) {
            sweep++;
            converged = 1;
            break;
        }
    }

    for (int j = 0; j < n; j++) {
        double norm_sq = 0.0;
        for (int k = 0; k < n; k++) {
            norm_sq += U_scaled[k * n + j] * U_scaled[k * n + j];
        }
        sigma[j] = sqrt(norm_sq);
    }

    if (sweeps_out) {
        *sweeps_out = converged ? sweep : max_sweeps;
    }
    return converged ? SOLVER_SUCCESS : SOLVER_NOT_CONVERGED;
}

/**
 * Direct Methods
 */

/* Gaussian Elimination with Partial Pivoting */
int gaussian_elimination(double *A, double *b, double *x, int n, solver_info *info) {
    // Define indexing macros for better readability
    #define AUG(row, col) (aug[(row) * (n + 1) + (col)])
    #define MATRIX_A(row, col) (A[(row) * n + (col)])

    int result = SOLVER_SUCCESS;

    // Allocate memory for an augmented matrix [A|b]
    double *aug = (double*)malloc(n * (n + 1) * sizeof(double));
    if (!aug) return SOLVER_MEMORY_ERROR;

    // Create augmented matrix [A|b]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AUG(i, j) = MATRIX_A(i, j);
        }
        AUG(i, n) = b[i];  // Last column contains the right-hand side vector
    }

    // Forward elimination with partial pivoting
    for (int k = 0; k < n - 1; k++) {
        // Find pivot element (partial pivoting)
        int pivot_row = k;
        double pivot_value = fabs(AUG(k, k));

        for (int i = k + 1; i < n; i++) {
            double current_value = fabs(AUG(i, k));
            if (current_value > pivot_value) {
                pivot_value = current_value;
                pivot_row = i;
            }
        }

        // Swap rows if a better pivot was found
        if (pivot_row != k) {
            for (int j = k; j <= n; j++) {
                double temp = AUG(k, j);
                AUG(k, j) = AUG(pivot_row, j);
                AUG(pivot_row, j) = temp;
            }
        }

        // Check for singularity (pivot too small)
        if (fabs(AUG(k, k)) < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }

        // Eliminate entries below the pivot
        for (int i = k + 1; i < n; i++) {
            double elimination_factor = AUG(i, k) / AUG(k, k);
            AUG(i, k) = 0.0;  // This element becomes zero

            // Update the rest of the row
            for (int j = k + 1; j <= n; j++) {
                AUG(i, j) -= elimination_factor * AUG(k, j);
            }
        }
    }

    // Check if the last pivot is too small (matrix is singular)
    if (fabs(AUG(n-1, n-1)) < TOLERANCE) {
        result = SOLVER_SINGULAR_MATRIX;
        goto cleanup;
    }

    // Back substitution to find the solution vector x
    for (int i = n - 1; i >= 0; i--) {
        // Start with the right-hand side value
        x[i] = AUG(i, n);

        // Subtract the effect of already computed unknowns
        for (int j = i + 1; j < n; j++) {
            x[i] -= AUG(i, j) * x[j];
        }

        // Divide by the diagonal coefficient
        x[i] /= AUG(i, i);
    }

cleanup:
    // Clean up allocated memory
    free(aug);

    // Remove macro definitions to avoid name conflicts
    #undef AUG
    #undef MATRIX_A

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Gauss-Jordan Elimination */
int gauss_jordan(double *A, double *b, double *x, int n, solver_info *info) {
    // Define indexing macro for better readability
    #define AUG(row, col) (aug[(row) * (n + 1) + (col)])

    int result = SOLVER_SUCCESS;

    // Allocate memory for an augmented matrix [A|b]
    double *aug = (double*)malloc(n * (n + 1) * sizeof(double));
    if (!aug) return SOLVER_MEMORY_ERROR;

    // Create augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AUG(i, j) = A[i * n + j];
        }
        AUG(i, n) = b[i];
    }

    // Gauss-Jordan elimination
    for (int k = 0; k < n; k++) {
        // Find pivot
        int max_row = k;
        double max_val = fabs(AUG(k, k));

        for (int i = k + 1; i < n; i++) {
            if (fabs(AUG(i, k)) > max_val) {
                max_val = fabs(AUG(i, k));
                max_row = i;
            }
        }

        // Swap rows if necessary
        if (max_row != k) {
            for (int j = 0; j <= n; j++) {
                double temp = AUG(k, j);
                AUG(k, j) = AUG(max_row, j);
                AUG(max_row, j) = temp;
            }
        }

        // Check for singularity
        if (fabs(AUG(k, k)) < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }

        // Scale row k
        double pivot = AUG(k, k);
        for (int j = k; j <= n; j++) {
            AUG(k, j) /= pivot;
        }

        // Eliminate all other rows
        for (int i = 0; i < n; i++) {
            if (i != k) {
                double factor = AUG(i, k);
                for (int j = k; j <= n; j++) {
                    AUG(i, j) -= factor * AUG(k, j);
                }
            }
        }
    }

    // Extract solution
    for (int i = 0; i < n; i++) {
        x[i] = AUG(i, n);
    }

cleanup:
    // Clean up memory
    free(aug);

    // Undefine macro to avoid name conflicts
    #undef AUG

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Back Substitution for upper triangular systems */
int back_substitution(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;

    // Perform back substitution for upper triangular matrix
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i * n + j] * x[j];
        }

        if (fabs(A[i * n + i]) < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }

        x[i] /= A[i * n + i];
    }

cleanup:
    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Forward Substitution for lower triangular systems */
int forward_substitution(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;

    // Perform forward substitution for lower triangular matrix
    for (int i = 0; i < n; i++) {
        x[i] = b[i];
        for (int j = 0; j < i; j++) {
            x[i] -= A[i * n + j] * x[j];
        }

        if (fabs(A[i * n + i]) < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }

        x[i] /= A[i * n + i];
    }

cleanup:
    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* LU Decomposition */
int lu_decomposition(double *A, double *b, double *x, int n, solver_info *info) {
    // Combined LU factors are stored in LU; partial pivoting yields PA = LU.
    double *LU = NULL;
    double *y = NULL;
    double *pb = NULL;
    int *piv = NULL;
    int result = SOLVER_SUCCESS;

    LU = (double*)malloc(n * n * sizeof(double));
    y = (double*)malloc(n * sizeof(double));
    pb = (double*)malloc(n * sizeof(double));
    piv = (int*)malloc(n * sizeof(int));

    if (!LU || !y || !pb || !piv) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    memcpy(LU, A, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // Doolittle factorization with partial pivoting
    for (int k = 0; k < n; k++) {
        int pivot_row = k;
        double pivot_value = fabs(LU[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double candidate = fabs(LU[i * n + k]);
            if (candidate > pivot_value) {
                pivot_value = candidate;
                pivot_row = i;
            }
        }

        if (pivot_value < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }

        if (pivot_row != k) {
            for (int j = 0; j < n; j++) {
                double temp = LU[k * n + j];
                LU[k * n + j] = LU[pivot_row * n + j];
                LU[pivot_row * n + j] = temp;
            }
            int temp_piv = piv[k];
            piv[k] = piv[pivot_row];
            piv[pivot_row] = temp_piv;
        }

        for (int i = k + 1; i < n; i++) {
            LU[i * n + k] /= LU[k * n + k];
            for (int j = k + 1; j < n; j++) {
                LU[i * n + j] -= LU[i * n + k] * LU[k * n + j];
            }
        }
    }

    // Apply the permutation to b, then forward/back substitute
    for (int i = 0; i < n; i++) {
        pb[i] = b[piv[i]];
    }

    for (int i = 0; i < n; i++) {
        y[i] = pb[i];
        for (int j = 0; j < i; j++) {
            y[i] -= LU[i * n + j] * y[j];
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= LU[i * n + j] * x[j];
        }
        x[i] /= LU[i * n + i];
    }

cleanup:
    free(LU);
    free(y);
    free(pb);
    free(piv);

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Cholesky Decomposition (for symmetric positive definite matrices) */
int cholesky(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *L = NULL;
    double *y = NULL;
    double *LT = NULL;

    // Check if matrix is symmetric
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (fabs(A[i * n + j] - A[j * n + i]) > TOLERANCE) {
                result = SOLVER_NOT_SPD_MATRIX;
                goto cleanup;
            }
        }
    }

    // Allocate memory for L and temporary vectors
    L = (double*)calloc(n * n, sizeof(double));
    y = (double*)malloc(n * sizeof(double));

    if (!L || !y) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Cholesky decomposition: A = L * L^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;

            if (j == i) { // Diagonal elements
                for (int k = 0; k < j; k++) {
                    sum += L[j * n + k] * L[j * n + k];
                }
                double val = A[j * n + j] - sum;

                if (val <= 0) {
                    result = SOLVER_NOT_SPD_MATRIX;
                    goto cleanup;
                }

                L[j * n + j] = sqrt(val);
            } else { // Non-diagonal elements
                for (int k = 0; k < j; k++) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }

    // Forward substitution Ly = b
    if (forward_substitution(L, b, y, n, NULL) != SOLVER_SUCCESS) {
        result = SOLVER_SINGULAR_MATRIX;
        goto cleanup;
    }

    // Transpose L for L^T
    LT = (double*)calloc(n * n, sizeof(double));
    if (!LT) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LT[i * n + j] = L[j * n + i];
        }
    }

    // Back substitution L^T x = y
    if (back_substitution(LT, y, x, n, NULL) != SOLVER_SUCCESS) {
        result = SOLVER_SINGULAR_MATRIX;
        goto cleanup;
    }

cleanup:
    // Clean up all allocated memory
    free(L);
    free(LT);
    free(y);

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* QR Decomposition */
int qr_decomposition(double *A, double *b, double *x, int n, solver_info *info) {
    // Allocate memory for all arrays at once
    double *Q = (double*)malloc(n * n * sizeof(double));
    double *R = (double*)calloc(n * n, sizeof(double));
    double *QT = NULL;  // Allocate later only if needed
    double *y = (double*)malloc(n * sizeof(double));
    double *u = NULL;   // Will be allocated/freed in the loop
    int result = SOLVER_SUCCESS;

    // Check memory allocation
    if (!Q || !R || !y) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Copy A to R initially
    memcpy(R, A, n * n * sizeof(double));

    // Initialize Q to identity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // QR decomposition using Householder reflections
    for (int k = 0; k < n - 1; k++) {
        // Compute Householder vector norm
        double alpha = 0.0;
        for (int i = k; i < n; i++) {
            alpha += R[i * n + k] * R[i * n + k];
        }
        alpha = sqrt(alpha);

        // Adjust sign of alpha based on diagonal element
        if (R[k * n + k] > 0) {
            alpha = -alpha;
        }

        // Calculate first element of Householder vector and its squared norm
        double u1 = R[k * n + k] - alpha;
        double norm_u_squared = u1 * u1;

        for (int i = k + 1; i < n; i++) {
            norm_u_squared += R[i * n + k] * R[i * n + k];
        }

        // Skip if the norm is too small (avoid division by near-zero)
        if (norm_u_squared < TOLERANCE) {
            continue;
        }

        // Calculate the Householder coefficient
        double beta = -2.0 / norm_u_squared;

        // Allocate memory for Householder vector
        u = (double*)calloc(n, sizeof(double));
        if (!u) {
            result = SOLVER_MEMORY_ERROR;
            goto cleanup;
        }

        // Fill Householder vector
        u[k] = u1;
        for (int i = k + 1; i < n; i++) {
            u[i] = R[i * n + k];
        }

        // Apply Householder reflection to R
        for (int j = k; j < n; j++) {
            double dot = 0.0;
            for (int i = k; i < n; i++) {
                dot += u[i] * R[i * n + j];
            }
            for (int i = k; i < n; i++) {
                R[i * n + j] += beta * u[i] * dot;
            }
        }

        // Update Q matrix
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int i = k; i < n; i++) {
                dot += u[i] * Q[j * n + i];
            }
            for (int i = k; i < n; i++) {
                Q[j * n + i] += beta * u[i] * dot;
            }
        }

        // Free Householder vector memory immediately after use
        free(u);
        u = NULL;
    }

    // Allocate memory for Q transpose
    QT = (double*)malloc(n * n * sizeof(double));
    if (!QT) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Transpose Q to get Q^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            QT[i * n + j] = Q[j * n + i];
        }
    }

    // Calculate y = Q^T * b
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += QT[i * n + j] * b[j];
        }
    }

    // Back substitution to solve Rx = y
    result = back_substitution(R, y, x, n, NULL);

cleanup:
    // Clean up all allocated memory
    free(Q);
    free(R);
    free(QT);
    free(y);
    free(u);  // Safe to call even if u is NULL

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Matrix Inversion Method */
int matrix_inversion(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *A_inv = NULL;
    double *A_copy = NULL;
    double *identity = NULL;

    // Allocate memory for matrices
    A_inv = (double*)malloc(n * n * sizeof(double));
    A_copy = (double*)malloc(n * n * sizeof(double));
    identity = (double*)malloc(n * n * sizeof(double));

    if (!A_inv || !A_copy || !identity) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Create identity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            identity[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Make a copy of A to preserve original matrix
    memcpy(A_copy, A, n * n * sizeof(double));

    // Find inverse of A using Gauss-Jordan elimination with n right-hand sides (identity matrix)
    // Iterate through each column of the identity matrix as a separate right-hand side
    for (int col = 0; col < n; col++) {
        double *rhs = (double*)malloc(n * sizeof(double));
        double *sol = (double*)malloc(n * sizeof(double));

        if (!rhs || !sol) {
            free(rhs);
            free(sol);
            result = SOLVER_MEMORY_ERROR;
            goto cleanup;
        }

        // Extract column from identity matrix
        for (int i = 0; i < n; i++) {
            rhs[i] = identity[i * n + col];
        }

        // Solve the system for this column
        result = gaussian_elimination(A_copy, rhs, sol, n, NULL);
        if (result != SOLVER_SUCCESS) {
            free(rhs);
            free(sol);
            goto cleanup;
        }

        // Place the solution into the appropriate column of A_inv
        for (int i = 0; i < n; i++) {
            A_inv[i * n + col] = sol[i];
        }

        free(rhs);
        free(sol);
    }

    // Compute x = A_inv * b
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        for (int j = 0; j < n; j++) {
            x[i] += A_inv[i * n + j] * b[j];
        }
    }

cleanup:
    // Clean up all allocated memory
    free(A_inv);
    free(A_copy);
    free(identity);

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Cramer's Rule Implementation */
int cramers_rule(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *modified_matrix = NULL;
    double det_A = 0.0;

    // For large matrices, Cramer's rule is inefficient
    if (n > 10) {
        set_info(info, 0, NAN, SOLVER_INVALID_PARAMETERS);
        return SOLVER_INVALID_PARAMETERS;
    }

    modified_matrix = (double*)malloc(n * n * sizeof(double));
    if (!modified_matrix) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Determinant of A via Gaussian elimination with partial pivoting
    det_A = matrix_determinant(A, n);

    // Check if A is singular (det(A) = 0)
    if (fabs(det_A) < TOLERANCE) {
        result = SOLVER_SINGULAR_MATRIX;
        goto cleanup;
    }

    // For each variable, replace the i-th column with b and take the ratio of determinants
    for (int i = 0; i < n; i++) {
        memcpy(modified_matrix, A, n * n * sizeof(double));
        for (int row = 0; row < n; row++) {
            modified_matrix[row * n + i] = b[row];
        }

        double det_modified = matrix_determinant(modified_matrix, n);
        x[i] = det_modified / det_A;
    }

cleanup:
    free(modified_matrix);
    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Row Reduction to Row-Echelon Form */
int row_echelon(double *A, double *b, double *x, int n, solver_info *info) {
    // Define indexing macro for better readability
    #define AUG(row, col) (aug[(row) * (n + 1) + (col)])

    int result = SOLVER_SUCCESS;

    // Allocate memory for an augmented matrix [A|b]
    double *aug = (double*)malloc(n * (n + 1) * sizeof(double));
    if (!aug) return SOLVER_MEMORY_ERROR;

    // Create augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AUG(i, j) = A[i * n + j];
        }
        AUG(i, n) = b[i];
    }

    // Forward phase: Convert to row echelon form
    int lead = 0;
    for (int r = 0; r < n; r++) {
        if (lead >= n) break;

        // Find the pivot row
        int i = r;
        while (i < n && fabs(AUG(i, lead)) < TOLERANCE) {
            i++;
        }

        if (i == n) {
            // No pivot in this column, move to next column
            lead++;
            r--; // Redo this row with new lead
            continue;
        }

        // Swap rows if necessary
        if (i != r) {
            for (int j = 0; j <= n; j++) {
                double temp = AUG(r, j);
                AUG(r, j) = AUG(i, j);
                AUG(i, j) = temp;
            }
        }

        // Scale the pivot row to have a leading 1
        double pivot = AUG(r, lead);
        if (fabs(pivot) < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }

        // Unlike reduced row echelon form, we don't normalize the pivot row
        // We just use it to eliminate entries below

        // Eliminate entries below the pivot
        for (int i = r + 1; i < n; i++) {
            double factor = AUG(i, lead) / pivot;
            for (int j = lead; j <= n; j++) {
                AUG(i, j) -= factor * AUG(r, j);
            }
        }

        lead++; // Move to the next column
    }

    // Check for consistency (zero rows with non-zero right side)
    for (int i = 0; i < n; i++) {
        int all_zeros = 1;
        for (int j = 0; j < n; j++) {
            if (fabs(AUG(i, j)) > TOLERANCE) {
                all_zeros = 0;
                break;
            }
        }
        if (all_zeros && fabs(AUG(i, n)) > TOLERANCE) {
            result = SOLVER_INCONSISTENT_SYSTEM;
            goto cleanup;
        }
    }

    // Back substitution to find the solution
    for (int i = n - 1; i >= 0; i--) {
        x[i] = AUG(i, n);
        for (int j = i + 1; j < n; j++) {
            x[i] -= AUG(i, j) * x[j];
        }
        if (fabs(AUG(i, i)) > TOLERANCE) {
            x[i] /= AUG(i, i);
        } else {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }
    }

cleanup:
    // Clean up memory
    free(aug);

    // Undefine macro to avoid name conflicts
    #undef AUG

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Reduced Row-Echelon Form */
int reduced_row_echelon(double *A, double *b, double *x, int n, solver_info *info) {
    // Define indexing macro for better readability
    #define AUG(row, col) (aug[(row) * (n + 1) + (col)])

    int result = SOLVER_SUCCESS;

    // Allocate memory for an augmented matrix [A|b]
    double *aug = (double*)malloc(n * (n + 1) * sizeof(double));
    if (!aug) {
        set_info(info, 0, NAN, SOLVER_MEMORY_ERROR);
        return SOLVER_MEMORY_ERROR;
    }

    // Create augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AUG(i, j) = A[i * n + j];
        }
        AUG(i, n) = b[i];
    }

    // Forward phase: Convert to row echelon form
    int lead = 0;
    int rank = 0;
    for (int r = 0; r < n; r++) {
        if (lead >= n) break;

        // Find the pivot row
        int i = r;
        while (i < n && fabs(AUG(i, lead)) < TOLERANCE) {
            i++;
        }

        if (i == n) {
            // No pivot in this column, move to next column
            lead++;
            r--; // Redo this row with new lead
            continue;
        }

        // Swap rows if necessary
        if (i != r) {
            for (int j = 0; j <= n; j++) {
                double temp = AUG(r, j);
                AUG(r, j) = AUG(i, j);
                AUG(i, j) = temp;
            }
        }

        // Scale the pivot row
        double pivot = AUG(r, lead);
        if (fabs(pivot) < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }

        for (int j = 0; j <= n; j++) {
            AUG(r, j) /= pivot;
        }

        // Eliminate current column entries in all other rows
        for (int i = 0; i < n; i++) {
            if (i != r) {
                double factor = AUG(i, lead);
                for (int j = 0; j <= n; j++) {
                    AUG(i, j) -= factor * AUG(r, j);
                }
            }
        }

        rank++;  // Recorded a pivot in this column
        lead++;  // Move to the next column
    }

    // Check for consistency (zero rows with non-zero right side)
    for (int i = 0; i < n; i++) {
        int all_zeros = 1;
        for (int j = 0; j < n; j++) {
            if (fabs(AUG(i, j)) > TOLERANCE) {
                all_zeros = 0;
                break;
            }
        }
        if (all_zeros && fabs(AUG(i, n)) > TOLERANCE) {
            result = SOLVER_INCONSISTENT_SYSTEM;
            goto cleanup;
        }
    }

    // A rank-deficient coefficient matrix has no unique solution; the row-index
    // extraction below would otherwise emit a bogus vector for skipped columns.
    if (rank < n) {
        result = SOLVER_SINGULAR_MATRIX;
        goto cleanup;
    }

    // Extract solution
    for (int i = 0; i < n; i++) {
        x[i] = AUG(i, n);
    }

cleanup:
    // Clean up memory
    free(aug);

    // Undefine macro to avoid name conflicts
    #undef AUG

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Triangularization method for solving linear systems */
int triangularization(double *A, double *b, double *x, int n, solver_info *info) {
    // Define indexing macros for better readability
    #define AUG(row, col) (aug[(row) * (n + 1) + (col)])

    int result = SOLVER_SUCCESS;

    // Allocate memory for an augmented matrix [A|b]
    double *aug = (double*)malloc(n * (n + 1) * sizeof(double));
    if (!aug) return SOLVER_MEMORY_ERROR;

    // Create augmented matrix [A|b]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AUG(i, j) = A[i * n + j];
        }
        AUG(i, n) = b[i];
    }

    // Forward elimination to create upper triangular matrix
    for (int k = 0; k < n - 1; k++) {
        // Find pivot element (partial pivoting)
        int pivot_row = k;
        double pivot_value = fabs(AUG(k, k));

        for (int i = k + 1; i < n; i++) {
            double current_value = fabs(AUG(i, k));
            if (current_value > pivot_value) {
                pivot_value = current_value;
                pivot_row = i;
            }
        }

        // Swap rows if a better pivot was found
        if (pivot_row != k) {
            for (int j = k; j <= n; j++) {
                double temp = AUG(k, j);
                AUG(k, j) = AUG(pivot_row, j);
                AUG(pivot_row, j) = temp;
            }
        }

        // Check for singularity (pivot too small)
        if (fabs(AUG(k, k)) < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }

        // Eliminate entries below the pivot
        for (int i = k + 1; i < n; i++) {
            double elimination_factor = AUG(i, k) / AUG(k, k);

            // Update the entire row
            for (int j = k; j <= n; j++) {
                AUG(i, j) -= elimination_factor * AUG(k, j);
            }
        }
    }

    // Check if the last pivot is too small (matrix is singular)
    if (fabs(AUG(n-1, n-1)) < TOLERANCE) {
        result = SOLVER_SINGULAR_MATRIX;
        goto cleanup;
    }

    // Back substitution phase to find the solution vector x
    for (int i = n - 1; i >= 0; i--) {
        // Start with the right-hand side value
        x[i] = AUG(i, n);

        // Subtract the effect of already computed unknowns
        for (int j = i + 1; j < n; j++) {
            x[i] -= AUG(i, j) * x[j];
        }

        // Divide by the diagonal coefficient
        x[i] /= AUG(i, i);
    }

cleanup:
    // Clean up allocated memory
    free(aug);

    // Remove macro definitions to avoid name conflicts
    #undef AUG

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/**
 * Iterative Methods
 */

/* Jacobi Iterative Method */
int jacobi(double *A, double *b, double *x, int n, solver_info *info) {
    double *x_new = (double*)malloc(n * sizeof(double));
    if (!x_new) {
        return SOLVER_MEMORY_ERROR;
    }

    // Initialize x with zeros
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    int iter;
    double residual;
    int result = SOLVER_SUCCESS;

    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Calculate new approximation
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum += A[i * n + j] * x[j];
                }
            }

            if (fabs(A[i * n + i]) < TOLERANCE) {
                result = SOLVER_SINGULAR_MATRIX;
                goto cleanup;
            }

            x_new[i] = (b[i] - sum) / A[i * n + i];
        }

        // Calculate residual and check convergence
        residual = 0.0;
        for (int i = 0; i < n; i++) {
            residual += fabs(x_new[i] - x[i]);
            x[i] = x_new[i];
        }

        if (residual < TOLERANCE) {
            break;
        }
    }

    // Update result if not converged
    if (iter == MAX_ITERATIONS) {
        result = SOLVER_NOT_CONVERGED;
    }

cleanup:
    free(x_new);
    set_info(info,
             (iter < MAX_ITERATIONS) ? iter + 1 : MAX_ITERATIONS,
             result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN,
             result);
    return result;
}

/* Gauss-Seidel Method */
int gauss_seidel(double *A, double *b, double *x, int n, solver_info *info) {
    int return_code = SOLVER_SUCCESS;

    // Initialize x with zeros
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    // Allocate memory for previous iteration values
    double *x_old = (double*)malloc(n * sizeof(double));
    if (!x_old) {
        return SOLVER_MEMORY_ERROR;
    }

    int iter;
    double residual = 0.0;

    // Main iteration loop
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Save current solution for convergence check
        memcpy(x_old, x, n * sizeof(double));

        // Gauss-Seidel iteration
        for (int i = 0; i < n; i++) {
            double sum1 = 0.0; // Sum of already computed x values
            double sum2 = 0.0; // Sum of not yet computed x values

            for (int j = 0; j < i; j++) {
                sum1 += A[i * n + j] * x[j];
            }

            for (int j = i + 1; j < n; j++) {
                sum2 += A[i * n + j] * x_old[j];
            }

            // Check for singular matrix
            if (fabs(A[i * n + i]) < TOLERANCE) {
                return_code = SOLVER_SINGULAR_MATRIX;
                goto cleanup;
            }

            x[i] = (b[i] - sum1 - sum2) / A[i * n + i];
        }

        // Calculate residual and check convergence
        residual = 0.0;
        for (int i = 0; i < n; i++) {
            residual += fabs(x[i] - x_old[i]);
        }

        if (residual < TOLERANCE) {
            break;
        }
    }

    // Check for convergence
    if (iter == MAX_ITERATIONS) {
        return_code = SOLVER_NOT_CONVERGED;
    }

cleanup:
    // Clean up allocated memory
    free(x_old);

    set_info(info,
             (iter < MAX_ITERATIONS) ? iter + 1 : MAX_ITERATIONS,
             return_code == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN,
             return_code);
    return return_code;
}

/* Successive Over-Relaxation Method */
int sor(double *A, double *b, double *x, int n, solver_info *info) {
    // Recommended omega value (relaxation factor) between 1.0 and 2.0
    double omega = 1.5;
    int result = SOLVER_SUCCESS;
    int iter = 0;
    double residual = 0.0;

    // Validate input parameters
    if (!A || !b || !x || n <= 0) {
        return SOLVER_INVALID_PARAM;
    }

    // Initialize x with zeros
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    // Allocate memory for previous solution
    double *x_old = (double*)malloc(n * sizeof(double));
    if (!x_old) {
        return SOLVER_MEMORY_ERROR;
    }

    // Main iteration loop
    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Save old solution
        memcpy(x_old, x, n * sizeof(double));

        // Update each component of x
        for (int i = 0; i < n; i++) {
            double sum1 = 0.0; // Sum of already computed x values
            double sum2 = 0.0; // Sum of not yet computed x values

            for (int j = 0; j < i; j++) {
                sum1 += A[i * n + j] * x[j];
            }

            for (int j = i + 1; j < n; j++) {
                sum2 += A[i * n + j] * x_old[j];
            }

            // Check for singular matrix
            if (fabs(A[i * n + i]) < TOLERANCE) {
                result = SOLVER_SINGULAR_MATRIX;
                goto cleanup;
            }

            // SOR formula
            x[i] = (1.0 - omega) * x_old[i] + (omega / A[i * n + i]) * (b[i] - sum1 - sum2);
        }

        // Calculate residual and check convergence
        residual = 0.0;
        for (int i = 0; i < n; i++) {
            residual += fabs(x[i] - x_old[i]);
        }

        if (residual < TOLERANCE) {
            break;
        }
    }

    // Check if we've reached maximum iterations
    if (iter == MAX_ITERATIONS) {
        result = SOLVER_NOT_CONVERGED;
    }

cleanup:
    // Clean up memory
    free(x_old);

    set_info(info,
             (iter < MAX_ITERATIONS) ? iter + 1 : MAX_ITERATIONS,
             result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN,
             result);
    return result;
}

/* Conjugate Gradient Method (for symmetric positive definite matrices) */
int conjugate_gradient(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    int iter = 0;
    double *r = NULL;
    double *p = NULL;
    double *Ap = NULL;

    int iterations_done = 0;

    // Check matrix symmetry
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (fabs(A[i * n + j] - A[j * n + i]) > TOLERANCE) {
                result = SOLVER_NOT_SPD_MATRIX;
                goto cleanup;
            }
        }
    }

    // Allocate memory for vectors
    r = (double*)malloc(n * sizeof(double));
    p = (double*)malloc(n * sizeof(double));
    Ap = (double*)malloc(n * sizeof(double));

    if (!r || !p || !Ap) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Initialize x with zeros
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    // Initial residual r = b - Ax
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        r[i] = b[i] - sum;
        p[i] = r[i]; // Initial search direction
    }

    double r_dot_r = 0.0;
    for (int i = 0; i < n; i++) {
        r_dot_r += r[i] * r[i];
    }

    if (sqrt(r_dot_r) < TOLERANCE) {
        result = SOLVER_SUCCESS;
        goto cleanup;
    }

    for (iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Calculate Ap
        for (int i = 0; i < n; i++) {
            Ap[i] = 0.0;
            for (int j = 0; j < n; j++) {
                Ap[i] += A[i * n + j] * p[j];
            }
        }

        // Calculate step size alpha
        double p_dot_Ap = 0.0;
        for (int i = 0; i < n; i++) {
            p_dot_Ap += p[i] * Ap[i];
        }

        // Convergence is checked first, so for an SPD matrix p^T A p > 0 here;
        // a non-positive value means A is not symmetric positive definite.
        if (p_dot_Ap <= 0.0) {
            result = SOLVER_NOT_SPD_MATRIX;
            iterations_done = iter + 1;
            goto cleanup;
        }

        double alpha = r_dot_r / p_dot_Ap;

        // Update solution and residual
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // Calculate new residual dot product
        double r_dot_r_new = 0.0;
        for (int i = 0; i < n; i++) {
            r_dot_r_new += r[i] * r[i];
        }

        // Check convergence
        if (sqrt(r_dot_r_new) < TOLERANCE) {
            result = SOLVER_SUCCESS;
            iterations_done = iter + 1;
            goto cleanup;
        }

        // Update search direction
        double beta = r_dot_r_new / r_dot_r;
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        r_dot_r = r_dot_r_new;
    }

    // Set result based on convergence
    iterations_done = iter;
    result = (iter == MAX_ITERATIONS) ? SOLVER_NOT_CONVERGED : SOLVER_SUCCESS;

cleanup:
    // Free all allocated memory
    free(r);
    free(p);
    free(Ap);

    set_info(info, iterations_done,
             result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Gradient Descent Method for solving SLEs */
int gradient_descent(double *A, double *b, double *x, int n, solver_info *info) {
    int iterations = 0;
    double residual_norm = 0.0;
    double *r = NULL;
    double *Ar = NULL;
    int result = SOLVER_SUCCESS;

    // Steepest descent for SPD systems requires a symmetric matrix.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (fabs(A[i * n + j] - A[j * n + i]) > TOLERANCE) {
                result = SOLVER_NOT_SPD_MATRIX;
                goto cleanup;
            }
        }
    }

    r = (double*)malloc(n * sizeof(double));
    Ar = (double*)malloc(n * sizeof(double));
    if (!r || !Ar) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Initialize x with zeros; residual r = b - Ax = b.
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        r[i] = b[i];
    }

    // Steepest descent with exact line search: alpha = (r.r) / (r.A.r).
    for (iterations = 0; iterations < MAX_ITERATIONS; iterations++) {
        double r_dot_r = 0.0;
        for (int i = 0; i < n; i++) {
            r_dot_r += r[i] * r[i];
        }
        residual_norm = sqrt(r_dot_r);
        if (residual_norm < TOLERANCE) {
            break;
        }

        for (int i = 0; i < n; i++) {
            Ar[i] = 0.0;
            for (int j = 0; j < n; j++) {
                Ar[i] += A[i * n + j] * r[j];
            }
        }

        double r_dot_Ar = 0.0;
        for (int i = 0; i < n; i++) {
            r_dot_Ar += r[i] * Ar[i];
        }

        // Convergence is checked first, so for an SPD matrix r^T A r > 0 here;
        // a non-positive value means A is not symmetric positive definite.
        if (r_dot_Ar <= 0.0) {
            result = SOLVER_NOT_SPD_MATRIX;
            goto cleanup;
        }

        double alpha = r_dot_r / r_dot_Ar;
        for (int i = 0; i < n; i++) {
            x[i] += alpha * r[i];
            r[i] -= alpha * Ar[i];
        }
    }

    if (iterations >= MAX_ITERATIONS && residual_norm >= TOLERANCE) {
        result = SOLVER_NO_CONVERGENCE;
    }

cleanup:
    free(r);
    free(Ar);

    set_info(info, iterations, result == SOLVER_SUCCESS ? residual_norm : NAN, result);
    return result;
}

/**
 * Minimal Residual Method (MINRES) for solving linear systems
 *
 * This method is particularly effective for symmetric indefinite systems,
 * but can also be used for non-symmetric systems.
 */
int minres(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    int steps = 0;

    // The symmetric Lanczos recurrence below is only valid for symmetric A.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (fabs(A[i * n + j] - A[j * n + i]) > TOLERANCE) {
                set_info(info, 0, NAN, SOLVER_NOT_SPD_MATRIX);
                return SOLVER_NOT_SPD_MATRIX;
            }
        }
    }

    // Lanczos tridiagonalization followed by a Givens least-squares solve, which
    // minimizes ||b - A x|| over the Krylov subspace (genuine minimal residual).
    double *V = NULL;      // Lanczos vectors, row k holds v_k (length n)
    double *w = NULL;      // work vector for A*v
    double *alpha = NULL;  // tridiagonal diagonal
    double *beta = NULL;   // tridiagonal off-diagonal (beta[0] = ||b||)
    double *H = NULL;      // (steps+1) x steps tridiagonal, reduced in place
    double *g = NULL;      // transformed right-hand side
    double *cs = NULL;     // Givens cosines
    double *sn = NULL;     // Givens sines
    double *y = NULL;      // least-squares solution coefficients

    V = (double*)calloc((size_t)(n + 1) * n, sizeof(double));
    w = (double*)malloc(n * sizeof(double));
    alpha = (double*)calloc(n, sizeof(double));
    beta = (double*)calloc(n + 1, sizeof(double));
    H = (double*)calloc((size_t)(n + 1) * n, sizeof(double));
    g = (double*)calloc(n + 1, sizeof(double));
    cs = (double*)calloc(n, sizeof(double));
    sn = (double*)calloc(n, sizeof(double));
    y = (double*)calloc(n, sizeof(double));

    if (!V || !w || !alpha || !beta || !H || !g || !cs || !sn || !y) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    double beta1 = 0.0;
    for (int i = 0; i < n; i++) {
        beta1 += b[i] * b[i];
    }
    beta1 = sqrt(beta1);

    if (beta1 < TOLERANCE) {
        result = SOLVER_SUCCESS;
        goto cleanup;
    }

    beta[0] = beta1;
    for (int i = 0; i < n; i++) {
        V[i] = b[i] / beta1;
    }

    for (steps = 0; steps < n; steps++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A[i * n + j] * V[steps * n + j];
            }
            w[i] = sum;
        }

        double a = 0.0;
        for (int i = 0; i < n; i++) {
            a += V[steps * n + i] * w[i];
        }
        alpha[steps] = a;

        for (int i = 0; i < n; i++) {
            w[i] -= a * V[steps * n + i];
            if (steps > 0) {
                w[i] -= beta[steps] * V[(steps - 1) * n + i];
            }
        }

        double bn = 0.0;
        for (int i = 0; i < n; i++) {
            bn += w[i] * w[i];
        }
        bn = sqrt(bn);
        beta[steps + 1] = bn;

        if (bn < TOLERANCE) {
            steps++;
            break;
        }
        for (int i = 0; i < n; i++) {
            V[(steps + 1) * n + i] = w[i] / bn;
        }
    }

    int m = steps;

    // Assemble the (m+1) x m tridiagonal least-squares system T_bar y = beta1 e1
    for (int j = 0; j < m; j++) {
        H[j * m + j] = alpha[j];
        H[(j + 1) * m + j] = beta[j + 1];
        if (j > 0) {
            H[(j - 1) * m + j] = beta[j];
        }
    }
    g[0] = beta1;

    // Reduce to upper triangular form with Givens rotations, transforming g alongside
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < j; i++) {
            double t1 = cs[i] * H[i * m + j] + sn[i] * H[(i + 1) * m + j];
            H[(i + 1) * m + j] = -sn[i] * H[i * m + j] + cs[i] * H[(i + 1) * m + j];
            H[i * m + j] = t1;
        }

        double h_jj = H[j * m + j];
        double h_j1j = H[(j + 1) * m + j];
        double denom = sqrt(h_jj * h_jj + h_j1j * h_j1j);
        if (denom < TOLERANCE) {
            cs[j] = 1.0;
            sn[j] = 0.0;
            continue;
        }
        cs[j] = h_jj / denom;
        sn[j] = h_j1j / denom;
        H[j * m + j] = denom;
        H[(j + 1) * m + j] = 0.0;

        double gt = cs[j] * g[j] + sn[j] * g[j + 1];
        g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
        g[j] = gt;
    }

    for (int i = m - 1; i >= 0; i--) {
        double sum = g[i];
        for (int k = i + 1; k < m; k++) {
            sum -= H[i * m + k] * y[k];
        }
        if (fabs(H[i * m + i]) < TOLERANCE) {
            result = SOLVER_NOT_CONVERGED;
            goto cleanup;
        }
        y[i] = sum / H[i * m + i];
    }

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < m; j++) {
            sum += V[j * n + i] * y[j];
        }
        x[i] = sum;
    }

cleanup:
    free(V);
    free(w);
    free(alpha);
    free(beta);
    free(H);
    free(g);
    free(cs);
    free(sn);
    free(y);
    set_info(info, steps, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/**
 * Generalized Minimal Residual Method (GMRES) for solving linear systems
 *
 * This method is particularly effective for solving non-symmetric linear systems.
 * It minimizes the residual norm over a Krylov subspace and requires matrix-vector
 * products only, making it suitable for large sparse systems.
 */
int gmres(double *A, double *b, double *x, int n, solver_info *info) {
    double *r = NULL;          // Residual vector
    double *v = NULL;          // Current basis vector
    double **V = NULL;         // Arnoldi basis vectors
    double **H = NULL;         // Hessenberg matrix
    double *cs = NULL;         // Givens rotation cosines
    double *sn = NULL;         // Givens rotation sines
    double *y = NULL;          // Solution of the least squares problem
    double *g = NULL;          // Right-hand side of the least-squares problem
    double *temp = NULL;       // Temporary vector for matrix-vector product

    double beta = 0.0;         // Initial residual norm
    double residual_norm = 0.0;// Current residual norm
    double initial_residual_norm = 0.0; // Initial residual norm
    int iter = 0;
    int total_iter = 0;
    int result = SOLVER_SUCCESS;
    int i, j;                  // Loop counters

    // GMRES restart parameter - could be made configurable
    int restart = n;

    // Allocate memory for vectors
    r = (double*)malloc(n * sizeof(double));
    v = (double*)malloc(n * sizeof(double));
    temp = (double*)malloc(n * sizeof(double));
    y = (double*)malloc((restart+1) * sizeof(double));
    g = (double*)malloc((restart+1) * sizeof(double));
    cs = (double*)malloc(restart * sizeof(double));
    sn = (double*)malloc(restart * sizeof(double));

    // Check if memory allocation was successful
    if (!r || !v || !temp || !y || !g || !cs || !sn) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Allocate memory for Arnoldi basis vectors
    V = (double**)malloc((restart+1) * sizeof(double*));
    if (!V) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Initialize all pointers to NULL for safer cleanup
    for (i = 0; i <= restart; i++) {
        V[i] = NULL;
    }

    // Allocate memory for each basis vector
    for (i = 0; i <= restart; i++) {
        V[i] = (double*)malloc(n * sizeof(double));
        if (!V[i]) {
            result = SOLVER_MEMORY_ERROR;
            goto cleanup;
        }
    }

    // Allocate memory for Hessenberg matrix
    H = (double**)malloc((restart+1) * sizeof(double*));
    if (!H) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Initialize all pointers to NULL for safer cleanup
    for (i = 0; i <= restart; i++) {
        H[i] = NULL;
    }

    // Allocate memory for each row of H
    for (i = 0; i <= restart; i++) {
        H[i] = (double*)calloc(restart, sizeof(double));
        if (!H[i]) {
            result = SOLVER_MEMORY_ERROR;
            goto cleanup;
        }
    }

    // Initialize solution vector x to zeros if not provided
    for (i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    // Main GMRES loop (with restarts)
    while (total_iter < MAX_ITERATIONS) {
        // Compute initial residual r = b - A*x
        for (i = 0; i < n; i++) {
            temp[i] = 0.0;
            for (j = 0; j < n; j++) {
                temp[i] += A[i * n + j] * x[j];
            }
            r[i] = b[i] - temp[i];
        }

        // Calculate initial residual norm
        beta = 0.0;
        for (i = 0; i < n; i++) {
            beta += r[i] * r[i];
        }
        beta = sqrt(beta);

        // Remember initial residual norm on the first iteration
        if (total_iter == 0) {
            initial_residual_norm = beta;

            if (initial_residual_norm < TOLERANCE) {
                // b is already close to zero, solution is x = 0
                residual_norm = initial_residual_norm;
                iter = 0;
                goto reporting;
            }
        }

        // Initialize the first basis vector
        for (i = 0; i < n; i++) {
            V[0][i] = r[i] / beta;
        }

        // Initialize right-hand side of the least-squares problem
        for (i = 0; i <= restart; i++) {
            g[i] = 0.0;
        }
        g[0] = beta;

        // Arnoldi iteration
        for (iter = 0; iter < restart && total_iter < MAX_ITERATIONS; iter++, total_iter++) {
            // Compute v = A * V[iter]
            for (i = 0; i < n; i++) {
                v[i] = 0.0;
                for (j = 0; j < n; j++) {
                    v[i] += A[i * n + j] * V[iter][j];
                }
            }

            // Modified Gram-Schmidt orthogonalization
            for (j = 0; j <= iter; j++) {
                H[j][iter] = 0.0;
                for (i = 0; i < n; i++) {
                    H[j][iter] += v[i] * V[j][i];
                }
                for (i = 0; i < n; i++) {
                    v[i] -= H[j][iter] * V[j][i];
                }
            }

            // Compute the (iter+1,iter) element of H
            H[iter+1][iter] = 0.0;
            for (i = 0; i < n; i++) {
                H[iter+1][iter] += v[i] * v[i];
            }
            H[iter+1][iter] = sqrt(H[iter+1][iter]);

            // Check for breakdown (orthogonalization breakdown)
            if (fabs(H[iter+1][iter]) < TOLERANCE) {
                iter++;  // Include this iteration
                break;
            }

            // Create the next basis vector
            for (i = 0; i < n; i++) {
                V[iter+1][i] = v[i] / H[iter+1][iter];
            }

            // Apply previous Givens rotations to the current column of H
            for (j = 0; j < iter; j++) {
                double temp = cs[j] * H[j][iter] + sn[j] * H[j+1][iter];
                H[j+1][iter] = -sn[j] * H[j][iter] + cs[j] * H[j+1][iter];
                H[j][iter] = temp;
            }

            // Compute and apply the current Givens rotation
            if (fabs(H[iter+1][iter]) > TOLERANCE) {
                double t = sqrt(H[iter][iter] * H[iter][iter] + H[iter+1][iter] * H[iter+1][iter]);
                cs[iter] = H[iter][iter] / t;
                sn[iter] = H[iter+1][iter] / t;

                // Apply Givens rotation to the Hessenberg matrix
                H[iter][iter] = t;
                H[iter+1][iter] = 0.0;

                // Apply Givens rotation to the right-hand side
                double temp = cs[iter] * g[iter] + sn[iter] * g[iter+1];
                g[iter+1] = -sn[iter] * g[iter] + cs[iter] * g[iter+1];
                g[iter] = temp;
            }

            // Calculate residual norm as |g[iter+1]|
            residual_norm = fabs(g[iter+1]);

            // Check for convergence
            if (residual_norm / initial_residual_norm < TOLERANCE) {
                break;
            }
        }

        // Solve the upper triangular system H(0:iter-1,0:iter-1) * y = g(0:iter-1)
        for (i = iter - 1; i >= 0; i--) {
            y[i] = g[i];
            for (j = i + 1; j < iter; j++) {
                y[i] -= H[i][j] * y[j];
            }
            if (fabs(H[i][i]) < TOLERANCE) {
                y[i] = 0.0;
            } else {
                y[i] /= H[i][i];
            }
        }

        // Update the solution x = x + V * y
        for (i = 0; i < n; i++) {
            for (j = 0; j < iter; j++) {
                x[i] += V[j][i] * y[j];
            }
        }

        // Check for convergence
        if (residual_norm / initial_residual_norm < TOLERANCE) {
            break;
        }

        // If we've reached the maximum number of iterations, exit
        if (total_iter >= MAX_ITERATIONS) {
            break;
        }
    }

reporting:
    if (total_iter >= MAX_ITERATIONS &&
        residual_norm / initial_residual_norm >= TOLERANCE) {
        result = SOLVER_NO_CONVERGENCE;
    }

cleanup:
    // Free allocated memory - check each pointer before freeing
    free(r);
    free(v);
    free(temp);
    free(y);
    free(g);
    free(cs);
    free(sn);

    // Free the 2D arrays
    if (V) {
        for (i = 0; i <= restart; i++) {
            free(V[i]);  // free(NULL) is safe in standard C
        }
        free(V);
    }

    if (H) {
        for (i = 0; i <= restart; i++) {
            free(H[i]);  // free(NULL) is safe in standard C
        }
        free(H);
    }

    set_info(info, total_iter, result == SOLVER_SUCCESS ? residual_norm : NAN, result);
    return result;
}

/* Biconjugate Gradient Method */
int bicg(double *A, double *b, double *x, int n, solver_info *info) {
    double *r = NULL;      // Residual vector
    double *r_tilde = NULL; // Shadow residual vector
    double *p = NULL;      // Search direction
    double *p_tilde = NULL; // Shadow search direction
    double *Ap = NULL;     // A * p
    double *temp = NULL;   // Temporary vector for calculations

    double rho = 0.0, rho_prev = 0.0, alpha = 0.0, beta = 0.0;
    double residual_norm = 0.0, p_tilde_Ap = 0.0;
    int iter = 0;
    int result = SOLVER_SUCCESS;

    // Allocate all memory at once
    double *memory_block = (double*)malloc(6 * n * sizeof(double));
    if (!memory_block) {
        return SOLVER_MEMORY_ERROR;
    }

    // Assign pointers to different sections of the allocated block
    r = memory_block;
    r_tilde = memory_block + n;
    p = memory_block + 2 * n;
    p_tilde = memory_block + 3 * n;
    Ap = memory_block + 4 * n;
    temp = memory_block + 5 * n;

    // Initialize solution vector if not provided
    for (int i = 0; i < n; i++) {
        if (isnan(x[i])) x[i] = 0.0;
    }

    // Calculate initial residual r = b - A*x
    // First calculate A*x
    for (int i = 0; i < n; i++) {
        temp[i] = 0.0;
        for (int j = 0; j < n; j++) {
            temp[i] += A[i * n + j] * x[j];
        }
    }

    // Then calculate r = b - A*x and initialize r_tilde, p, and p_tilde
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - temp[i];
        r_tilde[i] = r[i];
        p[i] = r[i];
        p_tilde[i] = r[i];
    }

    // Calculate initial residual norm
    residual_norm = 0.0;
    for (int i = 0; i < n; i++) {
        residual_norm += r[i] * r[i];
    }
    residual_norm = sqrt(residual_norm);

    // BiCG iteration
    for (iter = 0; iter < MAX_ITERATIONS && residual_norm > TOLERANCE; iter++) {
        // Calculate rho = (r_tilde, r)
        rho = 0.0;
        for (int i = 0; i < n; i++) {
            rho += r_tilde[i] * r[i];
        }

        if (fabs(rho) < 1e-30) {
            result = SOLVER_NO_CONVERGENCE;
            break;
        }

        if (iter > 0) {
            // Calculate beta = rho / rho_prev
            beta = rho / rho_prev;

            // Update search directions
            for (int i = 0; i < n; i++) {
                p[i] = r[i] + beta * p[i];
                p_tilde[i] = r_tilde[i] + beta * p_tilde[i];
            }
        }

        // Calculate Ap = A * p
        for (int i = 0; i < n; i++) {
            Ap[i] = 0.0;
            for (int j = 0; j < n; j++) {
                Ap[i] += A[i * n + j] * p[j];
            }
        }

        // Calculate alpha = rho / (p_tilde, Ap)
        p_tilde_Ap = 0.0;
        for (int i = 0; i < n; i++) {
            p_tilde_Ap += p_tilde[i] * Ap[i];
        }

        if (fabs(p_tilde_Ap) < 1e-30) {
            result = SOLVER_NO_CONVERGENCE;
            break;
        }

        alpha = rho / p_tilde_Ap;

        // Update solution x and residual r
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // Calculate A^T * p_tilde (transposed operation)
        for (int i = 0; i < n; i++) {
            temp[i] = 0.0;
            for (int j = 0; j < n; j++) {
                temp[i] += A[j * n + i] * p_tilde[j]; // Note the transposed indexing
            }
        }

        // Update shadow residual r_tilde
        for (int i = 0; i < n; i++) {
            r_tilde[i] -= alpha * temp[i];
        }

        // Save current rho for next iteration
        rho_prev = rho;

        // Calculate residual norm for convergence check
        residual_norm = 0.0;
        for (int i = 0; i < n; i++) {
            residual_norm += r[i] * r[i];
        }
        residual_norm = sqrt(residual_norm);
    }

    // A run that exhausts the iteration budget without converging has not succeeded
    if (result == SOLVER_SUCCESS && iter >= MAX_ITERATIONS && residual_norm > TOLERANCE) {
        result = SOLVER_NOT_CONVERGED;
    }

    // Free allocated memory
    free(memory_block);

    set_info(info, iter, result == SOLVER_SUCCESS ? residual_norm : NAN, result);
    return result;
}

/* Iterative Refinement Method */
int iterative_refinement(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    int iterations = 0;
    double residual_norm = 0.0;
    double *r = NULL;
    double *d = NULL;
    double *A_copy = NULL;

    // Allocate memory for residual vector, correction vector, and matrix copy
    r = (double*)malloc(n * sizeof(double));
    if (!r) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    d = (double*)malloc(n * sizeof(double));
    if (!d) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    A_copy = (double*)malloc(n * n * sizeof(double));
    if (!A_copy) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Create a copy of A for repeated solving
    memcpy(A_copy, A, n * n * sizeof(double));

    // Get initial solution using a direct method (e.g., Gaussian elimination)
    result = gaussian_elimination(A, b, x, n, NULL);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // Iterative refinement loop
    for (iterations = 0; iterations < MAX_ITERATIONS; iterations++) {
        // Compute residual r = b - A*x
        for (int i = 0; i < n; i++) {
            r[i] = b[i];
            for (int j = 0; j < n; j++) {
                r[i] -= A[i * n + j] * x[j];
            }
        }

        // Calculate residual norm
        residual_norm = 0.0;
        for (int i = 0; i < n; i++) {
            residual_norm += r[i] * r[i];
        }
        residual_norm = sqrt(residual_norm);

        // Check for convergence
        if (residual_norm < TOLERANCE) {
            break;
        }

        // Solve A*d = r to find correction d
        result = gaussian_elimination(A_copy, r, d, n, NULL);
        if (result != SOLVER_SUCCESS) {
            goto cleanup;
        }

        // Update solution: x = x + d
        for (int i = 0; i < n; i++) {
            x[i] += d[i];
        }
    }

    // Exhausting the iteration budget without meeting tolerance is not success.
    if (result == SOLVER_SUCCESS && iterations >= MAX_ITERATIONS && residual_norm >= TOLERANCE) {
        result = SOLVER_NOT_CONVERGED;
    }

cleanup:
    // Clean up allocated memory
    if (r) free(r);
    if (d) free(d);
    if (A_copy) free(A_copy);

    set_info(info, iterations, result == SOLVER_SUCCESS ? residual_norm : NAN, result);
    return result;
}

/**
 * Specialized Methods
 */

/* Normal Equations for Least Squares */
int normal_equations(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *AT = NULL;      /* Transpose of A */
    double *ATA = NULL;     /* A^T * A matrix */
    double *ATb = NULL;     /* A^T * b vector */

    // Allocate memory for intermediate matrices and vectors
    AT = (double*)malloc(n * n * sizeof(double));
    if (!AT) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    ATA = (double*)malloc(n * n * sizeof(double));
    if (!ATA) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    ATb = (double*)malloc(n * sizeof(double));
    if (!ATb) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Compute A^T (transpose of A)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AT[j * n + i] = A[i * n + j];
        }
    }

    // Compute A^T * A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ATA[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                ATA[i * n + j] += AT[i * n + k] * A[k * n + j];
            }
        }
    }

    // Compute A^T * b
    for (int i = 0; i < n; i++) {
        ATb[i] = 0.0;
        for (int j = 0; j < n; j++) {
            ATb[i] += AT[i * n + j] * b[j];
        }
    }

    // Solve the normal equations system ATA * x = ATb
    // We'll use LU decomposition for this part since it's efficient for square systems
    result = lu_decomposition(ATA, ATb, x, n, NULL);

    // If LU decomposition fails, try a more robust method
    if (result != SOLVER_SUCCESS) {
        result = gaussian_elimination(ATA, ATb, x, n, NULL);
    }

cleanup:
    // Free allocated memory - check each pointer before freeing
    if (AT) free(AT);
    if (ATA) free(ATA);
    if (ATb) free(ATb);

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Orthogonal Projection Method */
int orthogonal_projection(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *AT = NULL;        // Transpose of A
    double *ATA = NULL;       // A^T * A
    double *ATb = NULL;       // A^T * b

    // Allocate memory for temporary matrices and vectors
    AT = (double*)malloc(n * n * sizeof(double));
    if (!AT) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    ATA = (double*)malloc(n * n * sizeof(double));
    if (!ATA) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    ATb = (double*)malloc(n * sizeof(double));
    if (!ATb) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Compute A transpose (A^T)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AT[i * n + j] = A[j * n + i];
        }
    }

    // Compute A^T * A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ATA[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                ATA[i * n + j] += AT[i * n + k] * A[k * n + j];
            }
        }
    }

    // Compute A^T * b
    for (int i = 0; i < n; i++) {
        ATb[i] = 0.0;
        for (int j = 0; j < n; j++) {
            ATb[i] += AT[i * n + j] * b[j];
        }
    }

    // Solve the normal equations (A^T * A) * x = A^T * b
    // Using a direct method, such as Gauss-Jordan
    result = gauss_jordan(ATA, ATb, x, n, NULL);

cleanup:
    // Free all allocated memory
    if (AT) free(AT);
    if (ATA) free(ATA);
    if (ATb) free(ATb);

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Singular Value Decomposition (SVD) */
int svd(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *U_scaled = NULL;  // Columns scaled by the singular values
    double *V = NULL;         // Right singular vectors as columns
    double *S = NULL;         // Singular values
    int sweeps = 0;

    U_scaled = (double*)malloc(n * n * sizeof(double));
    V = (double*)malloc(n * n * sizeof(double));
    S = (double*)malloc(n * sizeof(double));
    if (!U_scaled || !V || !S) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    result = one_sided_jacobi_svd(A, n, U_scaled, V, S, &sweeps);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // x = V * S^+ * U^T * b, where column j of U_scaled equals sigma_j * u_j
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        if (S[j] > TOLERANCE) {
            double ub = 0.0;
            for (int k = 0; k < n; k++) {
                ub += U_scaled[k * n + j] * b[k];
            }
            double coeff = ub / (S[j] * S[j]);
            for (int k = 0; k < n; k++) {
                x[k] += V[k * n + j] * coeff;
            }
        }
    }

cleanup:
    free(U_scaled);
    free(V);
    free(S);
    set_info(info, sweeps, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Pseudoinverse Method using SVD */
int pseudoinverse(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *U_scaled = NULL;  // Columns scaled by the singular values
    double *V = NULL;         // Right singular vectors as columns
    double *S = NULL;         // Singular values
    double *A_pinv = NULL;    // Moore-Penrose pseudoinverse
    int sweeps = 0;

    U_scaled = (double*)malloc(n * n * sizeof(double));
    V = (double*)malloc(n * n * sizeof(double));
    S = (double*)malloc(n * sizeof(double));
    A_pinv = (double*)calloc(n * n, sizeof(double));
    if (!U_scaled || !V || !S || !A_pinv) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    result = one_sided_jacobi_svd(A, n, U_scaled, V, S, &sweeps);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // A+ = sum_j (1/sigma_j) v_j u_j^T, with column j of U_scaled equal to sigma_j * u_j
    for (int j = 0; j < n; j++) {
        if (S[j] > TOLERANCE) {
            double inv_sigma_sq = 1.0 / (S[j] * S[j]);
            for (int i = 0; i < n; i++) {
                double scaled = V[i * n + j] * inv_sigma_sq;
                for (int k = 0; k < n; k++) {
                    A_pinv[i * n + k] += scaled * U_scaled[k * n + j];
                }
            }
        }
    }

    // x = A+ * b
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        for (int j = 0; j < n; j++) {
            x[i] += A_pinv[i * n + j] * b[j];
        }
    }

cleanup:
    free(U_scaled);
    free(V);
    free(S);
    free(A_pinv);
    set_info(info, sweeps, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Block Matrix Method for solving linear systems */
int block_matrix(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;

    // Check for valid input dimensions
    if (n <= 1) {
        // For small matrices, use direct Gaussian elimination
        return gaussian_elimination(A, b, x, n, info);
    }

    // Choose block size (approximately half the matrix dimension)
    int block_size = n / 2;
    int remaining = n - block_size;

    // Allocate memory for the blocks and temporary vectors
    double *A11 = NULL; // Top-left block
    double *A12 = NULL; // Top-right block
    double *A21 = NULL; // Bottom-left block
    double *A22 = NULL; // Bottom-right block
    double *b1 = NULL;  // Top part of b
    double *b2 = NULL;  // Bottom part of b
    double *x1 = NULL;  // Top part of solution
    double *x2 = NULL;  // Bottom part of solution
    double *S = NULL;   // Schur complement
    double *temp = NULL; // Temporary vector
    double *Y = NULL;   // For solving A11 * Y = A12
    double *col = NULL; // For column-by-column operations

    // Allocate memory for all blocks and vectors
    A11 = (double*)malloc(block_size * block_size * sizeof(double));
    A12 = (double*)malloc(block_size * remaining * sizeof(double));
    A21 = (double*)malloc(remaining * block_size * sizeof(double));
    A22 = (double*)malloc(remaining * remaining * sizeof(double));
    b1 = (double*)malloc(block_size * sizeof(double));
    b2 = (double*)malloc(remaining * sizeof(double));
    x1 = (double*)malloc(block_size * sizeof(double));
    x2 = (double*)malloc(remaining * sizeof(double));
    S = (double*)malloc(remaining * remaining * sizeof(double));
    // Allocate temp to the maximum size needed (block_size or remaining)
    temp = (double*)malloc((block_size > remaining ? block_size : remaining) * sizeof(double));
    Y = (double*)malloc(block_size * remaining * sizeof(double));
    col = (double*)malloc(block_size * sizeof(double));

    // Check if memory allocation was successful
    if (!A11 || !A12 || !A21 || !A22 || !b1 || !b2 || !x1 || !x2 ||
        !S || !temp || !Y || !col) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Extract blocks from the original matrix A
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            A11[i * block_size + j] = A[i * n + j];
        }
        for (int j = 0; j < remaining; j++) {
            A12[i * remaining + j] = A[i * n + block_size + j];
        }
        b1[i] = b[i];
    }

    for (int i = 0; i < remaining; i++) {
        for (int j = 0; j < block_size; j++) {
            A21[i * block_size + j] = A[(block_size + i) * n + j];
        }
        for (int j = 0; j < remaining; j++) {
            A22[i * remaining + j] = A[(block_size + i) * n + block_size + j];
        }
        b2[i] = b[block_size + i];
    }

    // Solve A11 * y1 = b1 for y1 (using a direct method)
    result = gaussian_elimination(A11, b1, x1, block_size, NULL);
    if (result == SOLVER_SINGULAR_MATRIX) {
        // A singular leading block does not imply a singular full system; the
        // Schur-complement scheme cannot proceed, so solve the whole system
        // directly with partial pivoting.
        result = gaussian_elimination(A, b, x, n, NULL);
        goto cleanup;
    }
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // Compute temp = A21 * x1
    for (int i = 0; i < remaining; i++) {
        temp[i] = 0.0;
        for (int j = 0; j < block_size; j++) {
            temp[i] += A21[i * block_size + j] * x1[j];
        }
    }

    // Compute modified right-hand side for the second block: b2' = b2 - temp
    for (int i = 0; i < remaining; i++) {
        temp[i] = b2[i] - temp[i];
    }

    // Compute Schur complement: S = A22 - A21 * A11⁻¹ * A12
    // First, solve A11 * Y = A12 for Y column by column
    for (int j = 0; j < remaining; j++) {
        // Extract the j-th column of A12
        for (int i = 0; i < block_size; i++) {
            col[i] = A12[i * remaining + j];
        }

        // Solve A11 * y = col
        result = gaussian_elimination(A11, col, &Y[j * block_size], block_size, NULL);
        if (result != SOLVER_SUCCESS) {
            goto cleanup;
        }
    }

    // Compute S = A22 - A21 * Y
    for (int i = 0; i < remaining; i++) {
        for (int j = 0; j < remaining; j++) {
            double sum = 0.0;
            for (int k = 0; k < block_size; k++) {
                sum += A21[i * block_size + k] * Y[j * block_size + k];
            }
            S[i * remaining + j] = A22[i * remaining + j] - sum;
        }
    }

    // Solve S * x2 = temp for x2
    result = gaussian_elimination(S, temp, x2, remaining, NULL);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // Compute A12 * x2
    for (int i = 0; i < block_size; i++) {
        temp[i] = 0.0;
        for (int j = 0; j < remaining; j++) {
            temp[i] += A12[i * remaining + j] * x2[j];
        }
    }

    // Compute x1 = y1 - A11⁻¹ * A12 * x2
    // First, update temp = b1 - temp (where temp now contains A12 * x2)
    for (int i = 0; i < block_size; i++) {
        temp[i] = b1[i] - temp[i];
    }

    // Solve A11 * x1 = temp for the final x1
    result = gaussian_elimination(A11, temp, x1, block_size, NULL);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // Combine x1 and x2 into the final solution vector x
    for (int i = 0; i < block_size; i++) {
        x[i] = x1[i];
    }
    for (int i = 0; i < remaining; i++) {
        x[block_size + i] = x2[i];
    }

cleanup:
    // Clean up allocated memory
    free(A11);
    free(A12);
    free(A21);
    free(A22);
    free(b1);
    free(b2);
    free(x1);
    free(x2);
    free(S);
    free(temp);
    free(Y);
    free(col);

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Partitioning Method for solving linear systems */
int partitioning(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    int block_size = n / 2;  // Determine the block size (simple division for now)

    // Special case for small matrices
    if (n <= 3) {
        return gaussian_elimination(A, b, x, n, info);
    }

    // Define indexing macros for blocks
    #define A11(i, j) A[(i) * n + (j)]
    #define A12(i, j) A[(i) * n + (block_size + (j))]
    #define A21(i, j) A[(block_size + (i)) * n + (j)]
    #define A22(i, j) A[(block_size + (i)) * n + (block_size + (j))]

    // Declare all memory pointers initially as NULL
    double *b1 = NULL, *b2 = NULL, *x1 = NULL, *x2 = NULL;
    double *S = NULL, *temp = NULL, *A11_inv_A12 = NULL, *A11_matrix = NULL;
    double *A12_col = NULL, *result_col = NULL, *temp_vector = NULL;
    int second_block_size = n - block_size;

    // Allocate memory for blocks and intermediates
    b1 = (double*)malloc(block_size * sizeof(double));
    b2 = (double*)malloc(second_block_size * sizeof(double));
    x1 = (double*)malloc(block_size * sizeof(double));
    x2 = (double*)malloc(second_block_size * sizeof(double));
    S = (double*)malloc(second_block_size * second_block_size * sizeof(double));
    temp = (double*)malloc(block_size > second_block_size ?
                           block_size * sizeof(double) :
                           second_block_size * sizeof(double));

    // Check if memory allocation was successful
    if (!b1 || !b2 || !x1 || !x2 || !S || !temp) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Split the right-hand side
    for (int i = 0; i < block_size; i++) {
        b1[i] = b[i];
    }
    for (int i = 0; i < second_block_size; i++) {
        b2[i] = b[block_size + i];
    }

    // Create Schur complement S = A22 - A21 * A11^(-1) * A12

    // Allocate memory for A11^(-1) * A12 and A11 matrix
    A11_inv_A12 = (double*)malloc(block_size * second_block_size * sizeof(double));
    A11_matrix = (double*)malloc(block_size * block_size * sizeof(double));

    if (!A11_inv_A12 || !A11_matrix) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Extract A11 as a separate matrix
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            A11_matrix[i * block_size + j] = A11(i, j);
        }
    }

    // Allocate memory for column operations (outside the loop to reuse memory)
    A12_col = (double*)malloc(block_size * sizeof(double));
    result_col = (double*)malloc(block_size * sizeof(double));

    if (!A12_col || !result_col) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // For each column of A12, solve A11 * x = A12_col
    for (int j = 0; j < second_block_size; j++) {
        for (int i = 0; i < block_size; i++) {
            A12_col[i] = A12(i, j);
        }

        // Solve A11 * result_col = A12_col
        result = gaussian_elimination(A11_matrix, A12_col, result_col, block_size, NULL);
        if (result == SOLVER_SINGULAR_MATRIX) {
            // A singular leading block does not imply a singular full system;
            // fall back to solving the whole system with partial pivoting.
            result = gaussian_elimination(A, b, x, n, NULL);
            goto cleanup;
        }
        if (result != SOLVER_SUCCESS) {
            goto cleanup;
        }

        // Store the result
        for (int i = 0; i < block_size; i++) {
            A11_inv_A12[i * second_block_size + j] = result_col[i];
        }
    }

    // Compute S = A22 - A21 * A11_inv_A12
    for (int i = 0; i < second_block_size; i++) {
        for (int j = 0; j < second_block_size; j++) {
            S[i * second_block_size + j] = A22(i, j);

            // Subtract A21 * A11_inv_A12
            for (int k = 0; k < block_size; k++) {
                S[i * second_block_size + j] -= A21(i, k) * A11_inv_A12[k * second_block_size + j];
            }
        }
    }

    // We can now free A12_col and result_col as they're no longer needed
    free(A12_col);
    A12_col = NULL;
    free(result_col);
    result_col = NULL;

    // Allocate memory for temp_vector
    temp_vector = (double*)malloc(block_size * sizeof(double));
    if (!temp_vector) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Compute temporary vector temp = b2 - A21 * A11^(-1) * b1
    // First solve A11 * temp_vector = b1
    result = gaussian_elimination(A11_matrix, b1, temp_vector, block_size, NULL);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // Now compute temp = b2 - A21 * temp_vector
    for (int i = 0; i < second_block_size; i++) {
        temp[i] = b2[i];
        for (int j = 0; j < block_size; j++) {
            temp[i] -= A21(i, j) * temp_vector[j];
        }
    }

    // Free temp_vector as it's no longer needed
    free(temp_vector);
    temp_vector = NULL;

    // Solve Sx2 = temp
    result = gaussian_elimination(S, temp, x2, second_block_size, NULL);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // Compute x1 = A11^(-1) * (b1 - A12 * x2)
    for (int i = 0; i < block_size; i++) {
        temp[i] = b1[i];
        for (int j = 0; j < second_block_size; j++) {
            temp[i] -= A12(i, j) * x2[j];
        }
    }

    result = gaussian_elimination(A11_matrix, temp, x1, block_size, NULL);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // Combine x1 and x2 into the solution vector x
    for (int i = 0; i < block_size; i++) {
        x[i] = x1[i];
    }
    for (int i = 0; i < second_block_size; i++) {
        x[block_size + i] = x2[i];
    }

cleanup:
    // Clean up allocated memory - safe to free NULL pointers
    free(b1);
    free(b2);
    free(x1);
    free(x2);
    free(S);
    free(temp);
    free(A11_inv_A12);
    free(A11_matrix);
    free(A12_col);
    free(result_col);
    free(temp_vector);

    // Remove macro definitions
    #undef A11
    #undef A12
    #undef A21
    #undef A22

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Matrix Rank Method */
int matrix_rank(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *augmented = NULL;
    double *coef_matrix = NULL;
    double *temp_matrix = NULL;
    int *pivot_cols = NULL;

    // Allocate memory for augmented matrix [A|b]
    augmented = (double*)malloc(n * (n + 1) * sizeof(double));
    if (!augmented) return SOLVER_MEMORY_ERROR;

    // Create augmented matrix [A|b]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i * (n + 1) + j] = A[i * n + j];
        }
        augmented[i * (n + 1) + n] = b[i];
    }

    // Create a copy of coefficient matrix A for rank calculation
    coef_matrix = (double*)malloc(n * n * sizeof(double));
    if (!coef_matrix) {
        free(augmented);
        return SOLVER_MEMORY_ERROR;
    }

    // Copy A to coefficient matrix
    memcpy(coef_matrix, A, n * n * sizeof(double));

    // Calculate rank of coefficient matrix A
    int rank_A = 0;
    {
        // Convert to row echelon form
        temp_matrix = (double*)malloc(n * n * sizeof(double));
        if (!temp_matrix) {
            free(coef_matrix);
            free(augmented);
            return SOLVER_MEMORY_ERROR;
        }
        memcpy(temp_matrix, coef_matrix, n * n * sizeof(double));

        int lead = 0;
        for (int r = 0; r < n; r++) {
            if (lead >= n) break;

            int i = r;
            while (i < n && fabs(temp_matrix[i * n + lead]) < TOLERANCE) {
                i++;
            }

            if (i == n) {
                lead++;
                r--;
                continue;
            }

            // Swap rows i and r
            if (i != r) {
                for (int j = 0; j < n; j++) {
                    double temp = temp_matrix[r * n + j];
                    temp_matrix[r * n + j] = temp_matrix[i * n + j];
                    temp_matrix[i * n + j] = temp;
                }
            }

            // Scale the pivot row
            double pivot = temp_matrix[r * n + lead];
            for (int j = 0; j < n; j++) {
                temp_matrix[r * n + j] /= pivot;
            }

            // Eliminate other rows
            for (int i = 0; i < n; i++) {
                if (i != r) {
                    double factor = temp_matrix[i * n + lead];
                    for (int j = 0; j < n; j++) {
                        temp_matrix[i * n + j] -= factor * temp_matrix[r * n + j];
                    }
                }
            }

            lead++;
        }

        // Count non-zero rows to get the rank
        for (int i = 0; i < n; i++) {
            int is_zero_row = 1;
            for (int j = 0; j < n; j++) {
                if (fabs(temp_matrix[i * n + j]) > TOLERANCE) {
                    is_zero_row = 0;
                    break;
                }
            }
            if (!is_zero_row) {
                rank_A++;
            }
        }

        free(temp_matrix);
        temp_matrix = NULL;
    }

    // Calculate rank of augmented matrix [A|b]
    int rank_augmented = 0;
    {
        // Convert to row echelon form
        temp_matrix = (double*)malloc(n * (n + 1) * sizeof(double));
        if (!temp_matrix) {
            free(coef_matrix);
            free(augmented);
            return SOLVER_MEMORY_ERROR;
        }
        memcpy(temp_matrix, augmented, n * (n + 1) * sizeof(double));

        int lead = 0;
        for (int r = 0; r < n; r++) {
            if (lead >= n + 1) break;

            int i = r;
            while (i < n && fabs(temp_matrix[i * (n + 1) + lead]) < TOLERANCE) {
                i++;
            }

            if (i == n) {
                lead++;
                r--;
                continue;
            }

            // Swap rows i and r
            if (i != r) {
                for (int j = 0; j < n + 1; j++) {
                    double temp = temp_matrix[r * (n + 1) + j];
                    temp_matrix[r * (n + 1) + j] = temp_matrix[i * (n + 1) + j];
                    temp_matrix[i * (n + 1) + j] = temp;
                }
            }

            // Scale the pivot row
            double pivot = temp_matrix[r * (n + 1) + lead];
            for (int j = 0; j < n + 1; j++) {
                temp_matrix[r * (n + 1) + j] /= pivot;
            }

            // Eliminate other rows
            for (int i = 0; i < n; i++) {
                if (i != r) {
                    double factor = temp_matrix[i * (n + 1) + lead];
                    for (int j = 0; j < n + 1; j++) {
                        temp_matrix[i * (n + 1) + j] -= factor * temp_matrix[r * (n + 1) + j];
                    }
                }
            }

            lead++;
        }

        // Count non-zero rows to get the rank
        for (int i = 0; i < n; i++) {
            int is_zero_row = 1;
            for (int j = 0; j < n + 1; j++) {
                if (fabs(temp_matrix[i * (n + 1) + j]) > TOLERANCE) {
                    is_zero_row = 0;
                    break;
                }
            }
            if (!is_zero_row) {
                rank_augmented++;
            }
        }

        // Save the reduced matrix for solution
        memcpy(augmented, temp_matrix, n * (n + 1) * sizeof(double));
        free(temp_matrix);
        temp_matrix = NULL;
    }

    // Check if the system is consistent
    if (rank_A < rank_augmented) {
        // System is inconsistent (no solution)
        result = SOLVER_NO_SOLUTION;
        goto cleanup;
    }

    // Initialize solution vector to zeros
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    if (rank_A == n) {
        // Full rank - unique solution exists
        // Solve using back-substitution
        for (int i = n - 1; i >= 0; i--) {
            x[i] = augmented[i * (n + 1) + n]; // Start with right-hand side value
            for (int j = i + 1; j < n; j++) {
                x[i] -= augmented[i * (n + 1) + j] * x[j];
            }
            x[i] /= augmented[i * (n + 1) + i];
        }
    } else if (rank_A == rank_augmented) {
        // Rank deficient - infinitely many solutions
        // Find pivot variables and free variables
        pivot_cols = (int*)malloc(n * sizeof(int));
        if (!pivot_cols) {
            result = SOLVER_MEMORY_ERROR;
            goto cleanup;
        }

        // Initialize all variables as free
        for (int j = 0; j < n; j++) {
            pivot_cols[j] = -1; // -1 indicates a free variable
        }

        // Identify pivot columns
        int row = 0;
        for (int j = 0; j < n && row < rank_A; j++) {
            if (fabs(augmented[row * (n + 1) + j]) > TOLERANCE) {
                pivot_cols[j] = row;
                row++;
            }
        }

        // Set free variables to 0 and solve for pivot variables
        // Backward substitution
        for (int i = rank_A - 1; i >= 0; i--) {
            int pivot_col = 0;
            while (pivot_col < n && pivot_cols[pivot_col] != i) {
                pivot_col++;
            }

            if (pivot_col < n) {
                x[pivot_col] = augmented[i * (n + 1) + n];
                for (int j = pivot_col + 1; j < n; j++) {
                    x[pivot_col] -= augmented[i * (n + 1) + j] * x[j];
                }
                x[pivot_col] /= augmented[i * (n + 1) + pivot_col];
            }
        }

        result = SOLVER_MULTIPLE_SOLUTIONS;
    } else {
        // Should never reach here due to earlier check
        result = SOLVER_NO_SOLUTION;
    }

cleanup:
    if (pivot_cols) free(pivot_cols);
    if (temp_matrix) free(temp_matrix);
    if (coef_matrix) free(coef_matrix);
    if (augmented) free(augmented);

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Determinant Method (Cramer's Rule) */
int determinant(double *A, double *b, double *x, int n, solver_info *info) {
    // Validate input parameters
    if (!A || !b || !x || n <= 0) {
        set_info(info, 0, NAN, SOLVER_INPUT_ERROR);
        return SOLVER_INPUT_ERROR;
    }

    int result = SOLVER_SUCCESS;
    double det_A = 0.0;

    // Allocate memory for matrix copies
    double *A_copy = (double*)malloc(n * n * sizeof(double));
    double *temp_matrix = (double*)malloc(n * n * sizeof(double));

    if (!A_copy || !temp_matrix) {
        // Free any successfully allocated memory
        if (A_copy) free(A_copy);
        if (temp_matrix) free(temp_matrix);
        set_info(info, 0, NAN, SOLVER_MEMORY_ERROR);
        return SOLVER_MEMORY_ERROR;
    }

    // Copy the original matrix A
    memcpy(A_copy, A, n * n * sizeof(double));

    // Calculate determinant of A using Gaussian elimination
    double det = 1.0;

    // Gaussian elimination with partial pivoting
    for (int k = 0; k < n - 1; k++) {
        // Find pivot
        int pivot_row = k;
        double pivot_value = fabs(A_copy[k * n + k]);

        for (int i = k + 1; i < n; i++) {
            if (fabs(A_copy[i * n + k]) > pivot_value) {
                pivot_value = fabs(A_copy[i * n + k]);
                pivot_row = i;
            }
        }

        // Swap rows if needed
        if (pivot_row != k) {
            for (int j = k; j < n; j++) {
                double temp = A_copy[k * n + j];
                A_copy[k * n + j] = A_copy[pivot_row * n + j];
                A_copy[pivot_row * n + j] = temp;
            }
            // Each row exchange changes the determinant sign
            det = -det;
        }

        // Check for singularity
        if (fabs(A_copy[k * n + k]) < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }

        // Eliminate entries below pivot
        for (int i = k + 1; i < n; i++) {
            double factor = A_copy[i * n + k] / A_copy[k * n + k];
            A_copy[i * n + k] = 0.0;  // This element becomes zero

            for (int j = k + 1; j < n; j++) {
                A_copy[i * n + j] -= factor * A_copy[k * n + j];
            }
        }
    }

    // Calculate determinant as product of diagonal elements
    det_A = det;
    for (int i = 0; i < n; i++) {
        det_A *= A_copy[i * n + i];
    }

    // Check if matrix is singular (determinant close to zero)
    if (fabs(det_A) < TOLERANCE) {
        result = SOLVER_SINGULAR_MATRIX;
        goto cleanup;
    }

    // Calculate each x[i] using Cramer's rule
    for (int i = 0; i < n; i++) {
        // Copy the original matrix A to temp_matrix
        memcpy(temp_matrix, A, n * n * sizeof(double));

        // Replace the i-th column with vector b
        for (int j = 0; j < n; j++) {
            temp_matrix[j * n + i] = b[j];
        }

        // Calculate determinant of the modified matrix using Gaussian elimination
        double det_i = 1.0;

        // Gaussian elimination with partial pivoting
        for (int k = 0; k < n - 1; k++) {
            // Find pivot
            int pivot_row = k;
            double pivot_value = fabs(temp_matrix[k * n + k]);

            for (int j = k + 1; j < n; j++) {
                if (fabs(temp_matrix[j * n + k]) > pivot_value) {
                    pivot_value = fabs(temp_matrix[j * n + k]);
                    pivot_row = j;
                }
            }

            // Swap rows if needed
            if (pivot_row != k) {
                for (int j = k; j < n; j++) {
                    double temp = temp_matrix[k * n + j];
                    temp_matrix[k * n + j] = temp_matrix[pivot_row * n + j];
                    temp_matrix[pivot_row * n + j] = temp;
                }
                // Each row exchange changes the determinant sign
                det_i = -det_i;
            }

            // Check for singularity (should not happen if det_A is not zero)
            if (fabs(temp_matrix[k * n + k]) < TOLERANCE) {
                // Just set det_i to 0 and continue - we'll handle this at division time
                det_i = 0.0;
                break;
            }

            // Eliminate entries below pivot
            for (int j = k + 1; j < n; j++) {
                double factor = temp_matrix[j * n + k] / temp_matrix[k * n + k];
                temp_matrix[j * n + k] = 0.0;  // This element becomes zero

                for (int m = k + 1; m < n; m++) {
                    temp_matrix[j * n + m] -= factor * temp_matrix[k * n + m];
                }
            }
        }

        // Calculate determinant as product of diagonal elements
        for (int j = 0; j < n; j++) {
            det_i *= temp_matrix[j * n + j];
        }

        // Calculate the solution component
        x[i] = det_i / det_A;
    }

cleanup:
    // Free allocated memory
    free(A_copy);
    free(temp_matrix);

    set_info(info, 0, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/* Eigenvalue Decomposition Method */
int eigenvalue_decomposition(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    int sweeps = 0;

    // Cyclic Jacobi assumes a symmetric A; reject anything else rather than
    // silently returning a wrong spectrum.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (fabs(A[i * n + j] - A[j * n + i]) > TOLERANCE) {
                set_info(info, 0, NAN, SOLVER_NOT_SPD_MATRIX);
                return SOLVER_NOT_SPD_MATRIX;
            }
        }
    }

    // Symmetric eigensolver via cyclic Jacobi rotations: A = V * diag(lambda) * V^T
    double *M = (double*)malloc(n * n * sizeof(double));
    double *V = (double*)malloc(n * n * sizeof(double));
    double *y = (double*)malloc(n * sizeof(double));

    if (!M || !V || !y) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    memcpy(M, A, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    const int max_sweeps = 100;
    int converged = 0;
    for (sweeps = 0; sweeps < max_sweeps; sweeps++) {
        double off = 0.0;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                off += M[p * n + q] * M[p * n + q];
            }
        }
        if (off < 1e-24) {
            converged = 1;
            break;
        }

        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = M[p * n + q];
                if (apq == 0.0) {
                    continue;
                }
                double app = M[p * n + p];
                double aqq = M[q * n + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0.0)
                    ? 1.0 / (theta + sqrt(theta * theta + 1.0))
                    : -1.0 / (-theta + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0);
                double s = t * c;
                double h = t * apq;

                M[p * n + p] = app - h;
                M[q * n + q] = aqq + h;
                M[p * n + q] = 0.0;
                M[q * n + p] = 0.0;

                for (int i = 0; i < n; i++) {
                    if (i != p && i != q) {
                        double aip = M[i * n + p];
                        double aiq = M[i * n + q];
                        M[i * n + p] = c * aip - s * aiq;
                        M[p * n + i] = M[i * n + p];
                        M[i * n + q] = s * aip + c * aiq;
                        M[q * n + i] = M[i * n + q];
                    }
                }

                for (int i = 0; i < n; i++) {
                    double vip = V[i * n + p];
                    double viq = V[i * n + q];
                    V[i * n + p] = c * vip - s * viq;
                    V[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }

    if (!converged) {
        result = SOLVER_NOT_CONVERGED;
        goto cleanup;
    }

    // x = sum_i q_i (q_i^T b) / lambda_i, with q_i the i-th column of V and lambda_i = M[i][i]
    for (int i = 0; i < n; i++) {
        if (fabs(M[i * n + i]) < TOLERANCE) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }
        double dot = 0.0;
        for (int k = 0; k < n; k++) {
            dot += V[k * n + i] * b[k];
        }
        y[i] = dot / M[i * n + i];
    }

    for (int k = 0; k < n; k++) {
        x[k] = 0.0;
        for (int i = 0; i < n; i++) {
            x[k] += V[k * n + i] * y[i];
        }
    }

cleanup:
    free(M);
    free(V);
    free(y);
    set_info(info, sweeps, result == SOLVER_SUCCESS ? residual_norm(A, b, x, n) : NAN, result);
    return result;
}

/**
 * Solver Entry Point
 */

/* Main solver function that dispatches to the appropriate method */
int solve_linear_system(double *A, double *b, double *x, int n,
                        const char *method, solver_info *info) {
    // Validate inputs so C-level callers cannot trigger out-of-bounds access.
    if (!A || !b || !x || n <= 0) {
        set_info(info, 0, NAN, SOLVER_INPUT_ERROR);
        return SOLVER_INPUT_ERROR;
    }

    // Initialize info to a known state so that even a solver that returns early
    // without touching it leaves consistent diagnostics for the caller.
    set_info(info, 0, NAN, SOLVER_SUCCESS);

    // Default to Gaussian Elimination if no method specified
    if (!method || strlen(method) == 0) {
        method = "gaussian_elimination";
    }

    int result;

    // Dispatch to the appropriate solver method
    if (strcmp(method, "gaussian_elimination") == 0) {
        result = gaussian_elimination(A, b, x, n, info);
    } else if (strcmp(method, "gauss_jordan") == 0) {
        result = gauss_jordan(A, b, x, n, info);
    } else if (strcmp(method, "back_substitution") == 0) {
        result = back_substitution(A, b, x, n, info);
    } else if (strcmp(method, "forward_substitution") == 0) {
        result = forward_substitution(A, b, x, n, info);
    } else if (strcmp(method, "lu_decomposition") == 0) {
        result = lu_decomposition(A, b, x, n, info);
    } else if (strcmp(method, "cholesky") == 0) {
        result = cholesky(A, b, x, n, info);
    } else if (strcmp(method, "qr_decomposition") == 0) {
        result = qr_decomposition(A, b, x, n, info);
    } else if (strcmp(method, "matrix_inversion") == 0) {
        result = matrix_inversion(A, b, x, n, info);
    } else if (strcmp(method, "cramers_rule") == 0) {
        result = cramers_rule(A, b, x, n, info);
    } else if (strcmp(method, "row_echelon") == 0) {
        result = row_echelon(A, b, x, n, info);
    } else if (strcmp(method, "reduced_row_echelon") == 0) {
        result = reduced_row_echelon(A, b, x, n, info);
    } else if (strcmp(method, "triangularization") == 0) {
        result = triangularization(A, b, x, n, info);
    } else if (strcmp(method, "jacobi") == 0) {
        result = jacobi(A, b, x, n, info);
    } else if (strcmp(method, "gauss_seidel") == 0) {
        result = gauss_seidel(A, b, x, n, info);
    } else if (strcmp(method, "sor") == 0) {
        result = sor(A, b, x, n, info);
    } else if (strcmp(method, "conjugate_gradient") == 0) {
        result = conjugate_gradient(A, b, x, n, info);
    } else if (strcmp(method, "gradient_descent") == 0) {
        result = gradient_descent(A, b, x, n, info);
    } else if (strcmp(method, "minres") == 0) {
        result = minres(A, b, x, n, info);
    } else if (strcmp(method, "gmres") == 0) {
        result = gmres(A, b, x, n, info);
    } else if (strcmp(method, "bicg") == 0) {
        result = bicg(A, b, x, n, info);
    } else if (strcmp(method, "iterative_refinement") == 0) {
        result = iterative_refinement(A, b, x, n, info);
    } else if (strcmp(method, "normal_equations") == 0) {
        result = normal_equations(A, b, x, n, info);
    } else if (strcmp(method, "orthogonal_projection") == 0) {
        result = orthogonal_projection(A, b, x, n, info);
    } else if (strcmp(method, "svd") == 0) {
        result = svd(A, b, x, n, info);
    } else if (strcmp(method, "pseudoinverse") == 0) {
        result = pseudoinverse(A, b, x, n, info);
    } else if (strcmp(method, "block_matrix") == 0) {
        result = block_matrix(A, b, x, n, info);
    } else if (strcmp(method, "partitioning") == 0) {
        result = partitioning(A, b, x, n, info);
    } else if (strcmp(method, "matrix_rank") == 0) {
        result = matrix_rank(A, b, x, n, info);
    } else if (strcmp(method, "determinant") == 0) {
        result = determinant(A, b, x, n, info);
    } else if (strcmp(method, "eigenvalue_decomposition") == 0) {
        result = eigenvalue_decomposition(A, b, x, n, info);
    } else {
        // Method not recognized
        set_info(info, 0, NAN, SOLVER_INPUT_ERROR);
        return SOLVER_INPUT_ERROR;
    }

    // Success invariant: a reported success must actually solve A x = b. Guards
    // against methods (e.g. gmres/svd/pseudoinverse) reporting success with a
    // stale residual on inconsistent or singular systems.
    if (result == SOLVER_SUCCESS) {
        double bnorm = 0.0;
        for (int i = 0; i < n; i++) {
            bnorm += b[i] * b[i];
        }
        bnorm = sqrt(bnorm);
        double res = residual_norm(A, b, x, n);
        if (res > 1e-6 * (bnorm + 1.0)) {
            set_info(info, info ? info->iterations : 0, res, SOLVER_SINGULAR_MATRIX);
            return SOLVER_SINGULAR_MATRIX;
        }
    }

    // Guarantee info->error_code matches the returned value for C-level callers.
    if (info) {
        info->error_code = result;
    }
    return result;
}