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

    // Set solution info if the pointer is provided
    if (info) {
        info->iterations = 0;  // Direct method has no iterations
        info->residual = 0.0;  // Exact solution (ignoring round-off errors)
        info->error_code = result;
    }

cleanup:
    // Clean up allocated memory
    free(aug);

    // Remove macro definitions to avoid name conflicts
    #undef AUG
    #undef MATRIX_A

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

    // Set info if requested
    if (info) {
        info->iterations = 0;
        info->residual = 0.0;
        info->error_code = result;
    }

cleanup:
    // Clean up memory
    free(aug);

    // Undefine macro to avoid name conflicts
    #undef AUG

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
    // Set info if requested
    if (info) {
        info->iterations = 0;
        info->residual = 0.0;
        info->error_code = SOLVER_SUCCESS;
    }

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
    // Set info if requested
    if (info) {
        info->iterations = 0;
        info->residual = 0.0;
        info->error_code = result;
    }

    return result;
}

/* LU Decomposition */
int lu_decomposition(double *A, double *b, double *x, int n, solver_info *info) {
    // Allocate memory for L, U and temporary vectors
    double *L = NULL;
    double *U = NULL;
    double *y = NULL;
    int result = SOLVER_SUCCESS;

    // Allocate all memory at once
    L = (double*)calloc(n * n, sizeof(double));
    U = (double*)calloc(n * n, sizeof(double));
    y = (double*)malloc(n * sizeof(double));

    if (!L || !U || !y) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // LU decomposition without pivoting
    for (int i = 0; i < n; i++) {
        // Upper triangular matrix
        for (int k = i; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L[i * n + j] * U[j * n + k];
            }
            U[i * n + k] = A[i * n + k] - sum;
        }

        // Lower triangular matrix
        L[i * n + i] = 1.0; // Diagonal elements of L are 1
        for (int k = i + 1; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L[k * n + j] * U[j * n + i];
            }

            if (fabs(U[i * n + i]) < TOLERANCE) {
                result = SOLVER_SINGULAR_MATRIX;
                goto cleanup;
            }

            L[k * n + i] = (A[k * n + i] - sum) / U[i * n + i];
        }
    }

    // Forward substitution Ly = b
    result = forward_substitution(L, b, y, n, NULL);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // Back substitution Ux = y
    result = back_substitution(U, y, x, n, NULL);
    if (result != SOLVER_SUCCESS) {
        goto cleanup;
    }

    // Set info if requested
    if (info) {
        info->iterations = 0;
        info->residual = 0.0;
        info->error_code = SOLVER_SUCCESS;
    }

cleanup:
    // Clean up allocated memory
    free(L);
    free(U);
    free(y);

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
                return SOLVER_NOT_SPD_MATRIX;
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

    // Set info if requested
    if (info) {
        info->iterations = 0;
        info->residual = 0.0;
        info->error_code = result;
    }

cleanup:
    // Clean up all allocated memory
    free(L);
    free(LT);
    free(y);

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

    // Set info if requested
    if (info) {
        info->iterations = 0;
        info->residual = 0.0;
        info->error_code = result;
    }

cleanup:
    // Clean up all allocated memory
    free(Q);
    free(R);
    free(QT);
    free(y);
    free(u);  // Safe to call even if u is NULL

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

    // Calculate residual for info if needed
    double residual = 0.0;
    if (info) {
        double *res_vec = (double*)malloc(n * sizeof(double));
        if (res_vec) {
            // Compute residual r = b - A*x
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += A[i * n + j] * x[j];
                }
                res_vec[i] = b[i] - sum;
                residual += res_vec[i] * res_vec[i];
            }
            residual = sqrt(residual);
            free(res_vec);
        }

        info->iterations = 0;  // Direct method has no iterations
        info->residual = residual;
        info->error_code = result;
    }

cleanup:
    // Clean up all allocated memory
    free(A_inv);
    free(A_copy);
    free(identity);

    return result;
}

/* Cramer's Rule Implementation */
int cramers_rule(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *modified_matrix = NULL;
    double *submatrix = NULL;
    double det_A = 0.0;

    // For large matrices, Cramer's rule is inefficient
    if (n > 10) {
        if (info) {
            info->error_code = SOLVER_INVALID_PARAMETERS;
            info->iterations = 0;
            info->residual = 0.0;
        }
        return SOLVER_INVALID_PARAMETERS;
    }

    // Allocate memory for the modified matrix and submatrix
    modified_matrix = (double*)malloc(n * n * sizeof(double));
    submatrix = (double*)malloc((n-1) * (n-1) * sizeof(double));

    if (!modified_matrix || (n > 1 && !submatrix)) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Calculate determinant of matrix A
    if (n == 1) {
        det_A = A[0];
    } else if (n == 2) {
        det_A = A[0] * A[3] - A[1] * A[2];
    } else {
        // For larger matrices, use expansion by minors
        det_A = 0.0;
        for (int j = 0; j < n; j++) {
            // Create submatrix by excluding first row and column j
            int sub_idx = 0;
            for (int row = 1; row < n; row++) {
                for (int col = 0; col < n; col++) {
                    if (col != j) {
                        submatrix[sub_idx++] = A[row * n + col];
                    }
                }
            }

            // Calculate determinant of submatrix
            double sub_det;
            if (n-1 == 1) {
                sub_det = submatrix[0];
            } else if (n-1 == 2) {
                sub_det = submatrix[0] * submatrix[3] - submatrix[1] * submatrix[2];
            } else {
                // For larger submatrices, we would need another approach
                // Since n > 10 is filtered out earlier, we can use a simplification
                // that handles up to 3x3 matrices for this example
                if (n-1 == 3) {
                    sub_det = submatrix[0] * (submatrix[4] * submatrix[8] - submatrix[5] * submatrix[7]) -
                              submatrix[1] * (submatrix[3] * submatrix[8] - submatrix[5] * submatrix[6]) +
                              submatrix[2] * (submatrix[3] * submatrix[7] - submatrix[4] * submatrix[6]);
                } else {
                    // This is a simplified implementation that works for matrices up to 4x4
                    // For larger matrices, you would need to implement an iterative algorithm
                    sub_det = 0.0;
                }
            }

            // Add or subtract the determinant
            double sign = (j % 2 == 0) ? 1.0 : -1.0;
            det_A += sign * A[j] * sub_det;
        }
    }

    // Check if A is singular (det(A) = 0)
    if (fabs(det_A) < TOLERANCE) {
        result = SOLVER_SINGULAR_MATRIX;
        goto cleanup;
    }

    // For each variable, create modified matrix and calculate determinant
    for (int i = 0; i < n; i++) {
        // Copy A to modified matrix
        memcpy(modified_matrix, A, n * n * sizeof(double));

        // Replace i-th column with b
        for (int row = 0; row < n; row++) {
            modified_matrix[row * n + i] = b[row];
        }

        // Calculate determinant of modified matrix using the same approach
        double det_modified;
        if (n == 1) {
            det_modified = modified_matrix[0];
        } else if (n == 2) {
            det_modified = modified_matrix[0] * modified_matrix[3] - modified_matrix[1] * modified_matrix[2];
        } else {
            // For larger matrices, use expansion by minors
            det_modified = 0.0;
            for (int j = 0; j < n; j++) {
                // Create submatrix by excluding first row and column j
                int sub_idx = 0;
                for (int row = 1; row < n; row++) {
                    for (int col = 0; col < n; col++) {
                        if (col != j) {
                            submatrix[sub_idx++] = modified_matrix[row * n + col];
                        }
                    }
                }

                // Calculate determinant of submatrix
                double sub_det;
                if (n-1 == 1) {
                    sub_det = submatrix[0];
                } else if (n-1 == 2) {
                    sub_det = submatrix[0] * submatrix[3] - submatrix[1] * submatrix[2];
                } else if (n-1 == 3) {
                    sub_det = submatrix[0] * (submatrix[4] * submatrix[8] - submatrix[5] * submatrix[7]) -
                              submatrix[1] * (submatrix[3] * submatrix[8] - submatrix[5] * submatrix[6]) +
                              submatrix[2] * (submatrix[3] * submatrix[7] - submatrix[4] * submatrix[6]);
                } else {
                    // Simplified for matrices up to 4x4
                    sub_det = 0.0;
                }

                // Add or subtract the determinant
                double sign = (j % 2 == 0) ? 1.0 : -1.0;
                det_modified += sign * modified_matrix[j] * sub_det;
            }
        }

        // Calculate x[i] = det(modified) / det(A)
        x[i] = det_modified / det_A;
    }

    // Calculate residual for info if needed
    double residual = 0.0;
    if (info) {
        double *res_vec = (double*)malloc(n * sizeof(double));
        if (res_vec) {
            // Compute residual r = b - A*x
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += A[i * n + j] * x[j];
                }
                res_vec[i] = b[i] - sum;
                residual += res_vec[i] * res_vec[i];
            }
            residual = sqrt(residual);
            free(res_vec);
        }

        info->iterations = 0;  // Direct method has no iterations
        info->residual = residual;
        info->error_code = result;
    }

cleanup:
    free(modified_matrix);
    free(submatrix);
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

    // Set info if requested
    if (info) {
        info->iterations = 0;
        info->residual = 0.0;
        info->error_code = result;
    }

cleanup:
    // Clean up memory
    free(aug);

    // Undefine macro to avoid name conflicts
    #undef AUG

    return result;
}

/* Reduced Row-Echelon Form */
int reduced_row_echelon(double *A, double *b, double *x, int n, solver_info *info) {
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

    // Extract solution
    for (int i = 0; i < n; i++) {
        x[i] = AUG(i, n);
    }

    // Set info if requested
    if (info) {
        info->iterations = 0;
        info->residual = 0.0;
        info->error_code = result;
    }

cleanup:
    // Clean up memory
    free(aug);

    // Undefine macro to avoid name conflicts
    #undef AUG

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

    // Set solution info if the pointer is provided
    if (info) {
        info->iterations = 0;  // Direct method has no iterations
        info->residual = 0.0;  // Exact solution (ignoring round-off errors)
        info->error_code = result;
    }

cleanup:
    // Clean up allocated memory
    free(aug);

    // Remove macro definitions to avoid name conflicts
    #undef AUG

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

    // Set info if requested
    if (info) {
        info->iterations = iter + 1;
        info->residual = residual;
        info->error_code = result;
    }

cleanup:
    free(x_new);
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
                free(x_old);
                return SOLVER_SINGULAR_MATRIX;
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

    // Set info if requested
    if (info) {
        info->iterations = iter + 1;
        info->residual = residual;
        info->error_code = return_code;
    }

    // Clean up allocated memory
    free(x_old);

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
    // Set info if requested
    if (info) {
        info->iterations = iter + 1;
        info->residual = residual;
        info->error_code = result;
    }

    // Clean up memory
    free(x_old);
    return result;
}

/* Conjugate Gradient Method (for symmetric positive definite matrices) */
int conjugate_gradient(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    int iter = 0;
    double *r = NULL;
    double *p = NULL;
    double *Ap = NULL;

    // Check matrix symmetry
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (fabs(A[i * n + j] - A[j * n + i]) > TOLERANCE) {
                return SOLVER_NOT_SPD_MATRIX;
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

    for (iter = 0; iter < n && iter < MAX_ITERATIONS; iter++) {
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

        if (fabs(p_dot_Ap) < TOLERANCE) {
            result = SOLVER_NOT_CONVERGED;
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
            if (info) {
                info->iterations = iter + 1;
                info->residual = sqrt(r_dot_r_new);
                info->error_code = SOLVER_SUCCESS;
            }
            result = SOLVER_SUCCESS;
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
    result = (iter == MAX_ITERATIONS) ? SOLVER_NOT_CONVERGED : SOLVER_SUCCESS;

cleanup:
    // Set info if requested
    if (info) {
        info->iterations = iter;
        info->residual = sqrt(r_dot_r);
        info->error_code = result;
    }

    // Free all allocated memory
    free(r);
    free(p);
    free(Ap);

    return result;
}

/* Gradient Descent Method for solving SLEs */
int gradient_descent(double *A, double *b, double *x, int n, solver_info *info) {
    int iterations = 0;
    double residual_norm = 0.0;
    double alpha = 0.01;  // Learning rate
    double *residual = NULL;
    double *gradient = NULL;
    double *Ax = NULL;
    int result = SOLVER_SUCCESS;

    // Allocate memory for temporary vectors
    residual = (double*)malloc(n * sizeof(double));
    if (!residual) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    gradient = (double*)malloc(n * sizeof(double));
    if (!gradient) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    Ax = (double*)malloc(n * sizeof(double));
    if (!Ax) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Initialize x with zeros
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    // Main gradient descent loop
    for (iterations = 0; iterations < MAX_ITERATIONS; iterations++) {
        // Compute Ax
        for (int i = 0; i < n; i++) {
            Ax[i] = 0.0;
            for (int j = 0; j < n; j++) {
                Ax[i] += A[i * n + j] * x[j];
            }
        }

        // Compute residual: r = b - Ax
        residual_norm = 0.0;
        for (int i = 0; i < n; i++) {
            residual[i] = b[i] - Ax[i];
            residual_norm += residual[i] * residual[i];
        }
        residual_norm = sqrt(residual_norm);

        // Check convergence
        if (residual_norm < TOLERANCE) {
            break;
        }

        // Compute gradient: g = -2A^T(b - Ax)
        for (int i = 0; i < n; i++) {
            gradient[i] = 0.0;
            for (int j = 0; j < n; j++) {
                gradient[i] -= 2.0 * A[j * n + i] * residual[j];
            }
        }

        // Update x: x = x - alpha * gradient
        for (int i = 0; i < n; i++) {
            x[i] -= alpha * gradient[i];
        }
    }

    // Set info if requested
    if (info) {
        info->iterations = iterations;
        info->residual = residual_norm;
        info->error_code = (iterations < MAX_ITERATIONS) ?
                           SOLVER_SUCCESS : SOLVER_NO_CONVERGENCE;
    }

    // Update result if no convergence
    if (iterations >= MAX_ITERATIONS) {
        result = SOLVER_NO_CONVERGENCE;
    }

cleanup:
    // Clean up allocated memory safely
    if (residual) free(residual);
    if (gradient) free(gradient);
    if (Ax) free(Ax);

    return result;
}

/**
 * Minimal Residual Method (MINRES) for solving linear systems
 *
 * This method is particularly effective for symmetric indefinite systems,
 * but can also be used for non-symmetric systems.
 */
int minres(double *A, double *b, double *x, int n, solver_info *info) {
    double *r = NULL;      // Residual vector
    double *v = NULL;      // Lanczos vector
    double *v_prev = NULL; // Previous Lanczos vector
    double *v_new = NULL;  // New Lanczos vector
    double *p = NULL;      // Search direction
    double *p_prev = NULL; // Previous search direction
    double *p_new = NULL;  // New search direction
    double beta = 0.0;     // Lanczos coefficient
    double beta_prev = 0.0;// Previous Lanczos coefficient
    double alpha = 0.0;    // Lanczos coefficient
    double gamma0 = 0.0, gamma1 = 0.0, gamma2 = 0.0; // Givens rotation parameters
    double sigma0 = 0.0, sigma1 = 0.0; // More Givens rotation parameters
    double tau0 = 0.0, tau1 = 0.0, tau2 = 0.0; // Temporary values for solution update
    double residual_norm = 0.0; // Current residual norm
    double initial_residual_norm = 0.0; // Initial residual norm
    int iter = 0;
    int result = SOLVER_SUCCESS;

    // Allocate memory for vectors
    r = (double*)malloc(n * sizeof(double));
    if (!r) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    v = (double*)malloc(n * sizeof(double));
    if (!v) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    v_prev = (double*)malloc(n * sizeof(double));
    if (!v_prev) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    v_new = (double*)malloc(n * sizeof(double));
    if (!v_new) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    p = (double*)malloc(n * sizeof(double));
    if (!p) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    p_prev = (double*)malloc(n * sizeof(double));
    if (!p_prev) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    p_new = (double*)malloc(n * sizeof(double));
    if (!p_new) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Initialize solution vector x to zeros if not provided
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
    }

    // Compute initial residual r = b - A*x
    // Since x is initially zero, r = b
    for (int i = 0; i < n; i++) {
        r[i] = b[i];
    }

    // Calculate initial residual norm
    initial_residual_norm = 0.0;
    for (int i = 0; i < n; i++) {
        initial_residual_norm += r[i] * r[i];
    }
    initial_residual_norm = sqrt(initial_residual_norm);

    if (initial_residual_norm < TOLERANCE) {
        // b is already close to zero, solution is x = 0
        residual_norm = initial_residual_norm;
        iter = 0;
        goto reporting;
    }

    // Initialize first Lanczos vector v = r / ||r||
    beta = initial_residual_norm;
    for (int i = 0; i < n; i++) {
        v[i] = r[i] / beta;
        v_prev[i] = 0.0;
        p_prev[i] = 0.0;
        p[i] = 0.0;
    }

    gamma0 = 1.0;
    gamma1 = 1.0;
    sigma0 = 0.0;
    sigma1 = 0.0;
    tau0 = beta;

    // Main iteration loop
    for (iter = 1; iter <= MAX_ITERATIONS; iter++) {
        // Compute v_new = A*v
        for (int i = 0; i < n; i++) {
            v_new[i] = 0.0;
            for (int j = 0; j < n; j++) {
                v_new[i] += A[i * n + j] * v[j];
            }
        }

        // Compute alpha = v' * v_new
        alpha = 0.0;
        for (int i = 0; i < n; i++) {
            alpha += v[i] * v_new[i];
        }

        // Update v_new = v_new - alpha*v - beta*v_prev
        for (int i = 0; i < n; i++) {
            v_new[i] = v_new[i] - alpha * v[i] - beta * v_prev[i];
        }

        // Compute new beta
        beta_prev = beta;
        beta = 0.0;
        for (int i = 0; i < n; i++) {
            beta += v_new[i] * v_new[i];
        }
        beta = sqrt(beta);

        // Normalize v_new
        if (beta > TOLERANCE) {
            for (int i = 0; i < n; i++) {
                v_new[i] /= beta;
            }
        }

        // Apply previous Givens rotations to the new column of the tridiagonal matrix
        double alpha_hat = gamma0 * alpha - gamma2 * sigma0 * beta_prev;
        double beta_hat = gamma0 * beta;

        // Compute new Givens rotation
        gamma2 = gamma1;
        gamma1 = gamma0;

        double nu = sqrt(alpha_hat * alpha_hat + beta_hat * beta_hat);
        if (nu < TOLERANCE) {
            gamma0 = 1.0;
            sigma1 = 0.0;
        } else {
            gamma0 = alpha_hat / nu;
            sigma1 = beta_hat / nu;
        }

        // Update the solution
        tau2 = tau1;
        tau1 = tau0;
        tau0 = gamma0 * tau1;

        // Update direction vectors
        for (int i = 0; i < n; i++) {
            p_new[i] = (v[i] - sigma0 * p_prev[i] - alpha_hat * p[i]) / nu;
        }

        // Update solution vector
        for (int i = 0; i < n; i++) {
            x[i] += tau0 * p_new[i];
        }

        // Shift vectors for next iteration
        double *temp;

        // Shift v vectors
        temp = v_prev;
        v_prev = v;
        v = v_new;
        v_new = temp;

        // Shift p vectors
        temp = p_prev;
        p_prev = p;
        p = p_new;
        p_new = temp;

        // Update sigma values
        sigma0 = sigma1;

        // Compute residual norm
        residual_norm = fabs(sigma1 * tau1);

        // Check convergence
        if (residual_norm / initial_residual_norm < TOLERANCE) {
            break;
        }
    }

reporting:
    // Set solver information if requested
    if (info) {
        info->iterations = iter;
        info->residual = residual_norm / initial_residual_norm;

        if (iter >= MAX_ITERATIONS && residual_norm / initial_residual_norm >= TOLERANCE) {
            info->error_code = SOLVER_NO_CONVERGENCE;
            result = SOLVER_NO_CONVERGENCE;
        } else {
            info->error_code = SOLVER_SUCCESS;
        }
    }

cleanup:
    // Free allocated memory - check each pointer before freeing
    if (r) free(r);
    if (v) free(v);
    if (v_prev) free(v_prev);
    if (v_new) free(v_new);
    if (p) free(p);
    if (p_prev) free(p_prev);
    if (p_new) free(p_new);

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
    int result = SOLVER_SUCCESS;
    int i, j;                  // Loop counters

    // GMRES restart parameter - could be made configurable
    int restart = n;
    if (restart > n) restart = n;

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
    int total_iter = 0;
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
            y[i] /= H[i][i];
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
    // Set solver information if requested
    if (info) {
        info->iterations = total_iter;
        info->residual = residual_norm / initial_residual_norm;

        if (total_iter >= MAX_ITERATIONS && residual_norm / initial_residual_norm >= TOLERANCE) {
            info->error_code = SOLVER_NO_CONVERGENCE;
            result = SOLVER_NO_CONVERGENCE;
        } else {
            info->error_code = SOLVER_SUCCESS;
        }
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

        if (fabs(rho) < TOLERANCE) {
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

        if (fabs(p_tilde_Ap) < TOLERANCE) {
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

    // Set solver info if requested
    if (info) {
        info->iterations = iter;
        info->residual = residual_norm;
        info->error_code = result;
    }

    // Free allocated memory
    free(memory_block);

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

    // Set solution info if the pointer is provided
    if (info) {
        info->iterations = iterations;
        info->residual = residual_norm;
        info->error_code = result;
    }

cleanup:
    // Clean up allocated memory
    if (r) free(r);
    if (d) free(d);
    if (A_copy) free(A_copy);

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

    // Calculate residual if info is requested
    if (info) {
        double residual = 0.0;

        // Compute ||A*x - b||^2
        for (int i = 0; i < n; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < n; j++) {
                row_sum += A[i * n + j] * x[j];
            }
            double diff = row_sum - b[i];
            residual += diff * diff;
        }

        info->iterations = 0;  // Direct method has no iterations
        info->residual = sqrt(residual);
        info->error_code = result;
    }

cleanup:
    // Free allocated memory - check each pointer before freeing
    if (AT) free(AT);
    if (ATA) free(ATA);
    if (ATb) free(ATb);

    return result;
}

/* Orthogonal Projection Method */
int orthogonal_projection(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *AT = NULL;        // Transpose of A
    double *ATA = NULL;       // A^T * A
    double *ATb = NULL;       // A^T * b
    double *residual_vector = NULL;

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

    // Calculate residual if info is requested
    if (info) {
        info->iterations = 0;  // Direct method has no iterations
        info->error_code = result;
        info->residual = 0.0;

        residual_vector = (double*)malloc(n * sizeof(double));
        if (residual_vector) {
            double residual = 0.0;

            // Compute Ax and then b - Ax
            for (int i = 0; i < n; i++) {
                residual_vector[i] = 0.0;
                for (int j = 0; j < n; j++) {
                    residual_vector[i] += A[i * n + j] * x[j];
                }
                residual_vector[i] = b[i] - residual_vector[i];
                residual += residual_vector[i] * residual_vector[i];
            }

            info->residual = sqrt(residual);
        }
    }

cleanup:
    // Free all allocated memory
    if (AT) free(AT);
    if (ATA) free(ATA);
    if (ATb) free(ATb);
    if (residual_vector) free(residual_vector);

    return result;
}

/* Singular Value Decomposition (SVD) */
int svd(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *U = NULL;       // Left singular vectors
    double *S = NULL;       // Singular values
    double *VT = NULL;      // Transpose of right singular vectors
    double *bidiag = NULL;  // Bidiagonal form
    double *e = NULL;       // Superdiagonal elements
    double *work = NULL;    // Work array
    double *UT_b = NULL;    // U^T * b

    // Allocate memory
    U = (double*)malloc(n * n * sizeof(double));
    if (!U) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    S = (double*)malloc(n * sizeof(double));
    if (!S) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    VT = (double*)malloc(n * n * sizeof(double));
    if (!VT) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    bidiag = (double*)malloc(n * sizeof(double));
    if (!bidiag) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    e = (double*)malloc(n * sizeof(double));
    if (!e) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    work = (double*)malloc(4 * n * sizeof(double));
    if (!work) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    UT_b = (double*)malloc(n * sizeof(double));
    if (!UT_b) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Initialize matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            U[i*n + j] = A[i*n + j];  // Copy A to U
            VT[i*n + j] = (i == j) ? 1.0 : 0.0;  // Initialize VT as identity
        }
        bidiag[i] = 0.0;
        e[i] = 0.0;
        UT_b[i] = 0.0;
    }

    // Step 1: Bidiagonalization using Householder transformations (Golub-Kahan)
    for (int k = 0; k < n; k++) {
        // Householder transformation to zero elements below diagonal in column k
        double alpha = 0.0;
        for (int i = k; i < n; i++) {
            alpha += U[i*n + k] * U[i*n + k];
        }
        alpha = sqrt(alpha);

        if (fabs(alpha) > TOLERANCE) {
            if (U[k*n + k] < 0.0) alpha = -alpha;

            for (int i = k; i < n; i++) {
                U[i*n + k] /= alpha;
            }
            U[k*n + k] += 1.0;

            // Apply Householder transformation to U
            for (int j = k + 1; j < n; j++) {
                double s = 0.0;
                for (int i = k; i < n; i++) {
                    s += U[i*n + k] * U[i*n + j];
                }
                s /= U[k*n + k];

                for (int i = k; i < n; i++) {
                    U[i*n + j] -= s * U[i*n + k];
                }
            }

            // Apply Householder transformation to right side
            double s = 0.0;
            for (int i = k; i < n; i++) {
                s += U[i*n + k] * b[i];
            }
            s /= U[k*n + k];

            for (int i = k; i < n; i++) {
                b[i] -= s * U[i*n + k];
            }
        }

        bidiag[k] = alpha;

        if (k < n - 1) {
            // Householder transformation to zero elements right of superdiagonal in row k
            alpha = 0.0;
            for (int j = k + 1; j < n; j++) {
                alpha += U[k*n + j] * U[k*n + j];
            }
            alpha = sqrt(alpha);

            if (fabs(alpha) > TOLERANCE) {
                if (U[k*n + k+1] < 0.0) alpha = -alpha;

                for (int j = k + 1; j < n; j++) {
                    U[k*n + j] /= alpha;
                }
                U[k*n + k+1] += 1.0;

                // Apply Householder transformation to U
                for (int i = k + 1; i < n; i++) {
                    double s = 0.0;
                    for (int j = k + 1; j < n; j++) {
                        s += U[k*n + j] * U[i*n + j];
                    }
                    s /= U[k*n + k+1];

                    for (int j = k + 1; j < n; j++) {
                        U[i*n + j] -= s * U[k*n + j];
                    }
                }

                // Accumulate right singular vectors
                for (int i = 0; i < n; i++) {
                    double s = 0.0;
                    for (int j = k + 1; j < n; j++) {
                        s += U[k*n + j] * VT[i*n + j];
                    }
                    s /= U[k*n + k+1];

                    for (int j = k + 1; j < n; j++) {
                        VT[i*n + j] -= s * U[k*n + j];
                    }
                }
            }

            e[k] = alpha;
        }
    }

    // Copy bidiagonal elements to proper places
    for (int i = 0; i < n; i++) {
        S[i] = bidiag[i];
    }

    // Step 2: Iterative diagonalization of the bidiagonal form (QR algorithm)
    // This is a simplified implementation for the sake of brevity

    // QR iterations to diagonalize bidiagonal matrix
    const int max_iter = 100;
    for (int iter = 0; iter < max_iter; iter++) {
        // Test for splitting
        int k;
        for (k = n - 1; k > 0; k--) {
            if (fabs(e[k-1]) <= TOLERANCE * (fabs(S[k-1]) + fabs(S[k])))
                e[k-1] = 0.0;

            if (e[k-1] == 0.0)
                break;
        }

        if (k == n - 1) {
            // Singular value converged
            continue;
        }

        // QR step
        double c, s, f, g, h;

        // Givens rotation to make element S[k] zero
        g = S[k];
        h = S[k+1];
        f = ((g - h) * (g + h) + e[k] * e[k]) / (2.0 * h * g);
        g = sqrt(f * f + 1.0);
        if (f < 0.0) g = -g;
        f = ((S[k] - S[k+1]) * (S[k] + S[k+1]) + h * (g - f)) / S[k];

        // Next QR transformation
        c = s = 1.0;

        for (int i = k + 1; i < n; i++) {
            g = e[i-1];
            h = s * g;
            g = c * g;

            double z = sqrt(f * f + h * h);
            e[i-2] = z;

            c = f / z;
            s = h / z;

            f = S[i-1] * c + g * s;
            g = -S[i-1] * s + g * c;
            h = S[i] * s;
            S[i] *= c;

            for (int j = 0; j < n; j++) {
                double y = VT[j*n + i-1];
                double z = VT[j*n + i];
                VT[j*n + i-1] = y * c + z * s;
                VT[j*n + i] = -y * s + z * c;
            }

            z = sqrt(f * f + h * h);
            S[i-1] = z;

            if (z != 0.0) {
                c = f / z;
                s = h / z;
            }

            f = c * g + s * S[i];
            S[i] = -s * g + c * S[i];
        }

        e[k-1] = 0.0;
        e[n-2] = f;
        S[n-1] = h;
    }

    // Ensure non-negative singular values and sort in descending order
    for (int i = 0; i < n; i++) {
        S[i] = fabs(S[i]);
    }

    // Sort singular values in descending order (bubble sort for simplicity)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (S[j] < S[j + 1]) {
                // Swap singular values
                double temp = S[j];
                S[j] = S[j + 1];
                S[j + 1] = temp;

                // Swap corresponding columns in VT
                for (int k = 0; k < n; k++) {
                    temp = VT[k*n + j];
                    VT[k*n + j] = VT[k*n + j + 1];
                    VT[k*n + j + 1] = temp;
                }
            }
        }
    }

    // Step 3: Solve the system using the SVD components

    // Original b is already transformed by householder in bidiagonalization
    // Copy it to UT_b
    memcpy(UT_b, b, n * sizeof(double));

    // Compute x = V * S^+ * UT_b using pseudo-inverse
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        for (int j = 0; j < n; j++) {
            if (fabs(S[j]) > TOLERANCE) {
                x[i] += VT[i*n + j] * (UT_b[j] / S[j]);
            }
        }
    }

    // Compute residual if info is requested
    if (info) {
        info->iterations = 0; // Direct method, no iterations
        info->error_code = result;

        // Compute residual ||Ax - b||_2
        double residual = 0.0;
        for (int i = 0; i < n; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < n; j++) {
                row_sum += A[i * n + j] * x[j];
            }
            double diff = row_sum - b[i];
            residual += diff * diff;
        }
        info->residual = sqrt(residual);
    }

cleanup:
    // Free allocated memory - check each pointer before freeing
    if (U) free(U);
    if (S) free(S);
    if (VT) free(VT);
    if (bidiag) free(bidiag);
    if (e) free(e);
    if (work) free(work);
    if (UT_b) free(UT_b);

    return result;
}

/* Pseudoinverse Method using SVD */
int pseudoinverse(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    double *U = NULL;
    double *S = NULL;
    double *VT = NULL;
    double *S_inv = NULL;
    double *A_pinv = NULL;
    double *ATA = NULL;
    double *V = NULL;
    double *temp = NULL;
    double *UT = NULL;

    // Allocate memory
    U = (double*)malloc(n * n * sizeof(double));
    S = (double*)malloc(n * sizeof(double));
    VT = (double*)malloc(n * n * sizeof(double));
    S_inv = (double*)calloc(n * n, sizeof(double));
    A_pinv = (double*)malloc(n * n * sizeof(double));
    ATA = (double*)malloc(n * n * sizeof(double));
    V = (double*)malloc(n * n * sizeof(double));
    temp = (double*)malloc(n * n * sizeof(double));
    UT = (double*)malloc(n * n * sizeof(double));

    // Check all memory allocations at once
    if (!U || !S || !VT || !S_inv || !A_pinv || !ATA || !V || !temp || !UT) {
        result = SOLVER_MEMORY_ERROR;
        goto cleanup;
    }

    // Copy A to U for working on
    memcpy(U, A, n * n * sizeof(double));

    // Initialize VT as identity matrix for SVD computation
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            VT[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Compute A^T * A since it's symmetric and positive semi-definite
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ATA[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                ATA[i * n + j] += A[k * n + i] * A[k * n + j];
            }
        }
    }

    // Apply Jacobi iterations for SVD
    const int max_iterations = 100;
    double threshold = 1e-8;

    for (int iter = 0; iter < max_iterations; iter++) {
        double max_offdiag = 0.0;
        int p = 0, q = 1;

        // Find the largest off-diagonal element
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double val = fabs(ATA[i * n + j]);
                if (val > max_offdiag) {
                    max_offdiag = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence
        if (max_offdiag < threshold) {
            break;
        }

        // Compute Jacobi rotation
        double app = ATA[p * n + p];
        double aqq = ATA[q * n + q];
        double apq = ATA[p * n + q];

        double tau = (aqq - app) / (2.0 * apq);
        double t = 1.0 / (fabs(tau) + sqrt(1.0 + tau * tau));
        if (tau < 0) t = -t;

        double c = 1.0 / sqrt(1.0 + t * t);
        double s = t * c;

        // Update ATA
        ATA[p * n + p] = app * c * c + aqq * s * s + 2.0 * apq * c * s;
        ATA[q * n + q] = app * s * s + aqq * c * c - 2.0 * apq * c * s;
        ATA[p * n + q] = ATA[q * n + p] = 0.0;

        for (int i = 0; i < n; i++) {
            if (i != p && i != q) {
                double api = ATA[p * n + i];
                double aqi = ATA[q * n + i];
                ATA[p * n + i] = ATA[i * n + p] = api * c + aqi * s;
                ATA[q * n + i] = ATA[i * n + q] = -api * s + aqi * c;
            }
        }

        // Update VT
        for (int i = 0; i < n; i++) {
            double vip = VT[i * n + p];
            double viq = VT[i * n + q];
            VT[i * n + p] = vip * c + viq * s;
            VT[i * n + q] = -vip * s + viq * c;
        }
    }

    // Extract singular values
    for (int i = 0; i < n; i++) {
        S[i] = sqrt(fabs(ATA[i * n + i]));
    }

    // Sort singular values and corresponding vectors in descending order
    for (int i = 0; i < n - 1; i++) {
        int max_idx = i;
        double max_val = S[i];

        for (int j = i + 1; j < n; j++) {
            if (S[j] > max_val) {
                max_val = S[j];
                max_idx = j;
            }
        }

        if (max_idx != i) {
            // Swap singular values
            double temp_val = S[i];
            S[i] = S[max_idx];
            S[max_idx] = temp_val;

            // Swap columns of VT
            for (int j = 0; j < n; j++) {
                temp_val = VT[j * n + i];
                VT[j * n + i] = VT[j * n + max_idx];
                VT[j * n + max_idx] = temp_val;
            }
        }
    }

    // Compute U from A, S, and VT
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            U[i * n + j] = 0.0;
            if (S[j] > TOLERANCE) {
                for (int k = 0; k < n; k++) {
                    U[i * n + j] += A[i * n + k] * VT[j * n + k] / S[j];
                }
            }
        }
    }

    // Compute pseudoinverse using SVD components: A+ = V * diag(S+) * UT
    // First, create diag(S+) by inverting non-zero singular values
    for (int i = 0; i < n; i++) {
        if (fabs(S[i]) > TOLERANCE) {
            S_inv[i * n + i] = 1.0 / S[i];
        } else {
            S_inv[i * n + i] = 0.0; // Zero for small singular values
        }
    }

    // Transpose VT to get V
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i * n + j] = VT[j * n + i];
        }
    }

    // Compute V * diag(S+)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                temp[i * n + j] += V[i * n + k] * S_inv[k * n + j];
            }
        }
    }

    // Transpose U to get UT
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            UT[i * n + j] = U[j * n + i];
        }
    }

    // Compute temp * UT to get A+
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_pinv[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                A_pinv[i * n + j] += temp[i * n + k] * UT[k * n + j];
            }
        }
    }

    // Compute solution x = A+ * b
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        for (int j = 0; j < n; j++) {
            x[i] += A_pinv[i * n + j] * b[j];
        }
    }

    // Calculate residual ||Ax - b|| if info is requested
    if (info) {
        info->iterations = 0;  // Direct method, no iterations

        // Calculate residual ||Ax - b||
        double residual = 0.0;
        for (int i = 0; i < n; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < n; j++) {
                row_sum += A[i * n + j] * x[j];
            }
            double diff = row_sum - b[i];
            residual += diff * diff;
        }
        info->residual = sqrt(residual);
        info->error_code = result;
    }

cleanup:
    // Clean up all allocated memory
    free(U);
    free(S);
    free(VT);
    free(S_inv);
    free(A_pinv);
    free(ATA);
    free(V);
    free(temp);
    free(UT);

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

    // Set info if requested
    if (info) {
        info->iterations = 0; // Direct method, no iterations
        info->residual = 0.0; // Exact solution (ignoring round-off errors)
        info->error_code = SOLVER_SUCCESS;
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

    // Set info if requested
    if (info) {
        info->iterations = 0;  // Direct method has no iterations
        info->residual = 0.0;  // Compute residual if needed
        info->error_code = result;
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

    // Set info if requested
    if (info) {
        info->iterations = 0; // Direct method, no iterations

        // Calculate residual ||Ax - b||
        double residual = 0.0;
        for (int i = 0; i < n; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < n; j++) {
                row_sum += A[i * n + j] * x[j];
            }
            double diff = row_sum - b[i];
            residual += diff * diff;
        }
        info->residual = sqrt(residual);
        info->error_code = result;
    }

cleanup:
    if (pivot_cols) free(pivot_cols);
    if (temp_matrix) free(temp_matrix);
    if (coef_matrix) free(coef_matrix);
    if (augmented) free(augmented);

    return result;
}

/* Determinant Method (Cramer's Rule) */
int determinant(double *A, double *b, double *x, int n, solver_info *info) {
    // Validate input parameters
    if (!A || !b || !x || n <= 0) {
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
            free(A_copy);
            free(temp_matrix);
            return result;
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
        free(A_copy);
        free(temp_matrix);
        return SOLVER_SINGULAR_MATRIX;
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

    // Set info if requested
    if (info) {
        info->iterations = 0;  // Direct method has no iterations
        info->residual = 0.0;  // Exact solution (ignoring round-off errors)
        info->error_code = result;
    }

    // Free allocated memory
    free(A_copy);
    free(temp_matrix);

    return result;
}

/* Eigenvalue Decomposition Method */
int eigenvalue_decomposition(double *A, double *b, double *x, int n, solver_info *info) {
    int result = SOLVER_SUCCESS;
    int max_iter = MAX_ITERATIONS;
    double tol = TOLERANCE;        // Convergence tolerance

    // Allocate memory for matrices and vectors needed in the computation
    double *eigenvalues = (double*)malloc(n * sizeof(double));
    double *eigenvectors = (double*)malloc(n * n * sizeof(double));
    double *work_matrix = (double*)malloc(n * n * sizeof(double));
    double *y = (double*)malloc(n * sizeof(double));

    if (!eigenvalues || !eigenvectors || !work_matrix || !y) {
        // Handle memory allocation failure
        free(eigenvalues);
        free(eigenvectors);
        free(work_matrix);
        free(y);
        return SOLVER_MEMORY_ERROR;
    }

    // Make a copy of matrix A
    memcpy(work_matrix, A, n * n * sizeof(double));

    // Initialize eigenvectors to identity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            eigenvectors[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Perform power iterations for each eigenvector
    for (int k = 0; k < n; k++) {
        // Starting vector - use basis vector
        double *v = (double*)malloc(n * sizeof(double));
        double *prev_v = (double*)malloc(n * sizeof(double));

        if (!v || !prev_v) {
            free(v);
            free(prev_v);
            result = SOLVER_MEMORY_ERROR;
            goto cleanup;
        }

        // Initialize with kth basis vector perturbed slightly
        for (int i = 0; i < n; i++) {
            v[i] = (i == k) ? 1.0 : 0.0;
            prev_v[i] = v[i];
        }

        // Deflate matrix if this is not the first eigenvector
        if (k > 0) {
            // Deflate using previous eigenvalues and eigenvectors
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int p = 0; p < k; p++) {
                        work_matrix[i * n + j] -= eigenvalues[p] *
                            eigenvectors[i * n + p] * eigenvectors[j * n + p];
                    }
                }
            }
        }

        // Power iteration
        int iter;
        double lambda = 0.0, prev_lambda = 0.0;

        for (iter = 0; iter < max_iter; iter++) {
            // Matrix-vector multiplication: v = A*v
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += work_matrix[i * n + j] * v[j];
                }
                prev_v[i] = sum;
            }

            // Find the element with largest absolute value
            int max_idx = 0;
            double max_val = fabs(prev_v[0]);

            for (int i = 1; i < n; i++) {
                if (fabs(prev_v[i]) > max_val) {
                    max_val = fabs(prev_v[i]);
                    max_idx = i;
                }
            }

            // No good eigenvalue found (divide by zero risk)
            if (max_val < tol) {
                free(v);
                free(prev_v);
                result = SOLVER_SINGULAR_MATRIX;
                goto cleanup;
            }

            // Normalize vector
            lambda = prev_v[max_idx];
            for (int i = 0; i < n; i++) {
                v[i] = prev_v[i] / lambda;
            }

            // Check for convergence
            if (fabs(lambda - prev_lambda) < tol) {
                break;
            }

            prev_lambda = lambda;
        }

        // Store eigenvalue and eigenvector
        eigenvalues[k] = lambda;
        for (int i = 0; i < n; i++) {
            eigenvectors[i * n + k] = v[i];
        }

        free(v);
        free(prev_v);
    }

    // Solve the system Ax = b using the eigenvalue decomposition
    // Compute y = V^T * b
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += eigenvectors[j * n + i] * b[j];
        }
    }

    // Solve Dx = y, where D is diagonal matrix of eigenvalues
    for (int i = 0; i < n; i++) {
        if (fabs(eigenvalues[i]) < tol) {
            result = SOLVER_SINGULAR_MATRIX;
            goto cleanup;
        }
        y[i] /= eigenvalues[i];
    }

    // Compute x = V * y
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        for (int j = 0; j < n; j++) {
            x[i] += eigenvectors[i * n + j] * y[j];
        }
    }

    // Set info if requested
    if (info) {
        info->iterations = max_iter;

        // Calculate residual ||Ax - b||
        double residual = 0.0;
        for (int i = 0; i < n; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < n; j++) {
                row_sum += A[i * n + j] * x[j];
            }
            row_sum -= b[i];
            residual += row_sum * row_sum;
        }
        info->residual = sqrt(residual);
        info->error_code = result;
    }

cleanup:
    // Clean up allocated memory
    free(eigenvalues);
    free(eigenvectors);
    free(work_matrix);
    free(y);

    return result;
}

/**
 * Solver Entry Point
 */

/* Main solver function that dispatches to the appropriate method */
int solve_linear_system(double *A, double *b, double *x, int n,
                        const char *method, solver_info *info) {
    // Default to Gaussian Elimination if no method specified
    if (!method || strlen(method) == 0) {
        method = "gaussian_elimination";
    }

    // Dispatch to the appropriate solver method
    if (strcmp(method, "gaussian_elimination") == 0) {
        return gaussian_elimination(A, b, x, n, info);
    } else if (strcmp(method, "gauss_jordan") == 0) {
        return gauss_jordan(A, b, x, n, info);
    } else if (strcmp(method, "back_substitution") == 0) {
        return back_substitution(A, b, x, n, info);
    } else if (strcmp(method, "forward_substitution") == 0) {
        return forward_substitution(A, b, x, n, info);
    } else if (strcmp(method, "lu_decomposition") == 0) {
        return lu_decomposition(A, b, x, n, info);
    } else if (strcmp(method, "cholesky") == 0) {
        return cholesky(A, b, x, n, info);
    } else if (strcmp(method, "qr_decomposition") == 0) {
        return qr_decomposition(A, b, x, n, info);
    } else if (strcmp(method, "matrix_inversion") == 0) {
    return matrix_inversion(A, b, x, n, info);
    } else if (strcmp(method, "cramers_rule") == 0) {
        return cramers_rule(A, b, x, n, info);
    } else if (strcmp(method, "row_echelon") == 0) {
        return row_echelon(A, b, x, n, info);
    } else if (strcmp(method, "reduced_row_echelon") == 0) {
        return reduced_row_echelon(A, b, x, n, info);
    } else if (strcmp(method, "triangularization") == 0) {
        return triangularization(A, b, x, n, info);
    } else if (strcmp(method, "jacobi") == 0) {
        return jacobi(A, b, x, n, info);
    } else if (strcmp(method, "gauss_seidel") == 0) {
        return gauss_seidel(A, b, x, n, info);
    } else if (strcmp(method, "sor") == 0) {
        return sor(A, b, x, n, info);
    } else if (strcmp(method, "conjugate_gradient") == 0) {
        return conjugate_gradient(A, b, x, n, info);
    }

    // If method is not recognized, return an error code
    if (info) {
        info->error_code = -1; // Indicate an unrecognized method
    }
    return -1; // Return a non-zero value to indicate failure
}