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
#include "solver.h"

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

    // Function to calculate determinant inline
    double calculate_det(double *mat, int size) {
        // Base cases
        if (size == 1) return mat[0];
        if (size == 2) return mat[0] * mat[3] - mat[1] * mat[2];

        double det = 0.0;

        // Expand along the first row
        for (int j = 0; j < size; j++) {
            // Create submatrix by excluding first row and column j
            int sub_idx = 0;
            for (int row = 1; row < size; row++) {
                for (int col = 0; col < size; col++) {
                    if (col != j) {
                        submatrix[sub_idx++] = mat[row * size + col];
                    }
                }
            }

            // Add or subtract the determinant recursively
            double sign = (j % 2 == 0) ? 1.0 : -1.0;
            det += sign * mat[j] * calculate_det(submatrix, size-1);
        }

        return det;
    }

    // Calculate the determinant of A
    det_A = calculate_det(A, n);

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

        // Calculate determinant of modified matrix
        double det_modified = calculate_det(modified_matrix, n);

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