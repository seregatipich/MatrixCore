#ifndef SOLVERS_H
#define SOLVERS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Error code definitions */
#define SOLVER_SUCCESS              0
#define SOLVER_MEMORY_ERROR         1
#define SOLVER_SINGULAR_MATRIX      2
#define SOLVER_INVALID_PARAMETERS   3
#define SOLVER_INVALID_PARAM        3  /* Alias for SOLVER_INVALID_PARAMETERS */
#define SOLVER_NOT_CONVERGED        4
#define SOLVER_INCONSISTENT_SYSTEM  5
#define SOLVER_NOT_SPD_MATRIX       6
#define SOLVER_NO_CONVERGENCE       7
#define SOLVER_NO_SOLUTION          8
#define SOLVER_MULTIPLE_SOLUTIONS   9
#define SOLVER_INPUT_ERROR          10

/* Define solver information struct */
typedef struct {
    int iterations;       /* Number of iterations performed */
    double residual;      /* Final residual norm */
    int error_code;       /* Error code (0 for success, non-zero for failure) */
} solver_info;

/* Individual solver function prototypes */
int gaussian_elimination(double *A, double *b, double *x, int n, solver_info *info);
int gauss_jordan(double *A, double *b, double *x, int n, solver_info *info);
int back_substitution(double *A, double *b, double *x, int n, solver_info *info);
int forward_substitution(double *A, double *b, double *x, int n, solver_info *info);
int lu_decomposition(double *A, double *b, double *x, int n, solver_info *info);
int cholesky(double *A, double *b, double *x, int n, solver_info *info);
int qr_decomposition(double *A, double *b, double *x, int n, solver_info *info);
int matrix_inversion(double *A, double *b, double *x, int n, solver_info *info);
int cramers_rule(double *A, double *b, double *x, int n, solver_info *info);
int row_echelon(double *A, double *b, double *x, int n, solver_info *info);
int reduced_row_echelon(double *A, double *b, double *x, int n, solver_info *info);
int triangularization(double *A, double *b, double *x, int n, solver_info *info);
int jacobi(double *A, double *b, double *x, int n, solver_info *info);
int gauss_seidel(double *A, double *b, double *x, int n, solver_info *info);
int sor(double *A, double *b, double *x, int n, solver_info *info);
int conjugate_gradient(double *A, double *b, double *x, int n, solver_info *info);
int gradient_descent(double *A, double *b, double *x, int n, solver_info *info);
int minres(double *A, double *b, double *x, int n, solver_info *info);
int gmres(double *A, double *b, double *x, int n, solver_info *info);
int bicg(double *A, double *b, double *x, int n, solver_info *info);
int iterative_refinement(double *A, double *b, double *x, int n, solver_info *info);
int normal_equations(double *A, double *b, double *x, int n, solver_info *info);
int orthogonal_projection(double *A, double *b, double *x, int n, solver_info *info);
int svd(double *A, double *b, double *x, int n, solver_info *info);
int pseudoinverse(double *A, double *b, double *x, int n, solver_info *info);
int block_matrix(double *A, double *b, double *x, int n, solver_info *info);
int partitioning(double *A, double *b, double *x, int n, solver_info *info);
int matrix_rank(double *A, double *b, double *x, int n, solver_info *info);
int determinant(double *A, double *b, double *x, int n, solver_info *info);
int eigenvalue_decomposition(double *A, double *b, double *x, int n, solver_info *info);

/* Main solver function that dispatches to the appropriate method */
int solve_linear_system(double *A, double *b, double *x, int n, const char *method, solver_info *info);

#ifdef __cplusplus
}
#endif

#endif /* SOLVERS_H */