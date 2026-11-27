#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "mesh.hpp"

namespace krylov {

using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

/**
 * Build first derivative operator T_N for non-uniform mesh.
 * Uses second-order accurate finite differences in interior,
 * first-order at boundaries.
 *
 * Following Section 3 of Niesen & Wright (2011).
 */
inline SpMat build_first_derivative(const Mesh1D& mesh) {
    int N = mesh.N;
    std::vector<Triplet> triplets;
    triplets.reserve(3 * N);

    // First point: first-order forward difference
    triplets.emplace_back(0, 0, -1.0 / mesh.dx(0));
    triplets.emplace_back(0, 1, 1.0 / mesh.dx(0));

    // Interior points: second-order central difference
    for (int i = 1; i < N - 1; ++i) {
        double dx_im1 = mesh.dx(i - 1);  // x[i] - x[i-1]
        double dx_i = mesh.dx(i);         // x[i+1] - x[i]
        double sum = dx_im1 + dx_i;

        double beta_m1 = -dx_i / (dx_im1 * sum);
        double beta_0 = (dx_i - dx_im1) / (dx_im1 * dx_i);
        double beta_p1 = dx_im1 / (dx_i * sum);

        triplets.emplace_back(i, i - 1, beta_m1);
        triplets.emplace_back(i, i, beta_0);
        triplets.emplace_back(i, i + 1, beta_p1);
    }

    // Last point: first-order backward difference
    triplets.emplace_back(N - 1, N - 2, -1.0 / mesh.dx(N - 2));
    triplets.emplace_back(N - 1, N - 1, 1.0 / mesh.dx(N - 2));

    SpMat T(N, N);
    T.setFromTriplets(triplets.begin(), triplets.end());
    return T;
}

/**
 * Build second derivative operator S_N for non-uniform mesh.
 * Uses second-order accurate finite differences.
 * Boundary rows are zero (linear boundary conditions).
 */
inline SpMat build_second_derivative(const Mesh1D& mesh) {
    int N = mesh.N;
    std::vector<Triplet> triplets;
    triplets.reserve(3 * N);

    // Boundary points: zero (linear boundary condition)
    // First and last rows are all zeros

    // Interior points: second-order central difference
    for (int i = 1; i < N - 1; ++i) {
        double dx_im1 = mesh.dx(i - 1);
        double dx_i = mesh.dx(i);
        double sum = dx_im1 + dx_i;

        double gamma_m1 = 2.0 / (dx_im1 * sum);
        double gamma_0 = -2.0 / (dx_im1 * dx_i);
        double gamma_p1 = 2.0 / (dx_i * sum);

        triplets.emplace_back(i, i - 1, gamma_m1);
        triplets.emplace_back(i, i, gamma_0);
        triplets.emplace_back(i, i + 1, gamma_p1);
    }

    SpMat S(N, N);
    S.setFromTriplets(triplets.begin(), triplets.end());
    return S;
}

/**
 * Kronecker product for sparse matrices.
 * (A \otimes B)_{(i1*n2+i2), (j1*n2+j2)} = A_{i1,j1} * B_{i2,j2}
 */
inline SpMat kronecker_product(const SpMat& A, const SpMat& B) {
    int m1 = A.rows(), n1 = A.cols();
    int m2 = B.rows(), n2 = B.cols();

    std::vector<Triplet> triplets;
    triplets.reserve(A.nonZeros() * B.nonZeros());

    for (int k1 = 0; k1 < A.outerSize(); ++k1) {
        for (SpMat::InnerIterator it1(A, k1); it1; ++it1) {
            for (int k2 = 0; k2 < B.outerSize(); ++k2) {
                for (SpMat::InnerIterator it2(B, k2); it2; ++it2) {
                    int row = it1.row() * m2 + it2.row();
                    int col = it1.col() * n2 + it2.col();
                    triplets.emplace_back(row, col, it1.value() * it2.value());
                }
            }
        }
    }

    SpMat C(m1 * m2, n1 * n2);
    C.setFromTriplets(triplets.begin(), triplets.end());
    return C;
}

/**
 * Create identity matrix as sparse matrix.
 */
inline SpMat sparse_identity(int n) {
    SpMat I(n, n);
    I.setIdentity();
    return I;
}

/**
 * Build the full discretization matrix A for the 3D option pricing PDE.
 *
 * du/dt = (1/2) * sum_{i,j} rho_ij * sigma_i * sigma_j * s_i * s_j * d2u/ds_i*ds_j
 *       + r * sum_i s_i * du/ds_i - r * u
 *
 * This creates a sparse matrix of size (N1*N2*N3) x (N1*N2*N3).
 */
class OptionPDEOperator {
public:
    SpMat A;
    int N;

    OptionPDEOperator(const Mesh3D& mesh, double r,
                      const Eigen::Vector3d& sigma,
                      const Eigen::Matrix3d& rho) {

        int N1 = mesh.N1, N2 = mesh.N2, N3 = mesh.N3;
        N = N1 * N2 * N3;

        // Build 1D operators
        SpMat T1 = build_first_derivative(mesh.mesh1);
        SpMat T2 = build_first_derivative(mesh.mesh2);
        SpMat T3 = build_first_derivative(mesh.mesh3);

        SpMat S1 = build_second_derivative(mesh.mesh1);
        SpMat S2 = build_second_derivative(mesh.mesh2);
        SpMat S3 = build_second_derivative(mesh.mesh3);

        SpMat I1 = sparse_identity(N1);
        SpMat I2 = sparse_identity(N2);
        SpMat I3 = sparse_identity(N3);

        // Build diagonal matrices for s_i coefficients
        // D1 = diag(s1_values), etc.
        SpMat D1(N1, N1), D2(N2, N2), D3(N3, N3);
        {
            std::vector<Triplet> t1, t2, t3;
            for (int i = 0; i < N1; ++i) t1.emplace_back(i, i, mesh.mesh1.points(i));
            for (int i = 0; i < N2; ++i) t2.emplace_back(i, i, mesh.mesh2.points(i));
            for (int i = 0; i < N3; ++i) t3.emplace_back(i, i, mesh.mesh3.points(i));
            D1.setFromTriplets(t1.begin(), t1.end());
            D2.setFromTriplets(t2.begin(), t2.end());
            D3.setFromTriplets(t3.begin(), t3.end());
        }

        // Initialize A as zero
        A.resize(N, N);
        std::vector<Triplet> triplets;
        triplets.reserve(27 * N); // rough estimate

        // We'll build terms and accumulate them

        // Second derivative terms: (1/2) * rho_ij * sigma_i * sigma_j * s_i * s_j * d2u/ds_i*ds_j
        // Diagonal terms (i == j):
        // (1/2) * sigma_1^2 * s_1^2 * d2u/ds_1^2 => (I3 \otimes I2 \otimes D1^2 S1)
        // etc.

        // For mixed derivatives (i != j):
        // sigma_i * sigma_j * rho_ij * s_i * s_j * d2u/ds_i ds_j
        // d2u/ds_1 ds_2 = T1 \otimes T2 applied appropriately

        // Build each term and add to A
        double coeff11 = 0.5 * sigma(0) * sigma(0);
        double coeff22 = 0.5 * sigma(1) * sigma(1);
        double coeff33 = 0.5 * sigma(2) * sigma(2);
        double coeff12 = sigma(0) * sigma(1) * rho(0, 1);
        double coeff13 = sigma(0) * sigma(2) * rho(0, 2);
        double coeff23 = sigma(1) * sigma(2) * rho(1, 2);

        // Term 1: (1/2) * sigma_1^2 * s_1^2 * d2u/ds_1^2
        // = I3 \otimes I2 \otimes (D1^2 * S1)
        SpMat D1_sq = D1 * D1;
        SpMat term1 = kronecker_product(I3, kronecker_product(I2, D1_sq * S1));
        add_scaled_sparse(triplets, term1, coeff11);

        // Term 2: (1/2) * sigma_2^2 * s_2^2 * d2u/ds_2^2
        SpMat D2_sq = D2 * D2;
        SpMat term2 = kronecker_product(I3, kronecker_product(D2_sq * S2, I1));
        add_scaled_sparse(triplets, term2, coeff22);

        // Term 3: (1/2) * sigma_3^2 * s_3^2 * d2u/ds_3^2
        SpMat D3_sq = D3 * D3;
        SpMat term3 = kronecker_product(D3_sq * S3, kronecker_product(I2, I1));
        add_scaled_sparse(triplets, term3, coeff33);

        // Mixed term 12: rho_12 * sigma_1 * sigma_2 * s_1 * s_2 * d2u/ds_1 ds_2
        // = I3 \otimes (D2 * T2) \otimes (D1 * T1)
        SpMat term12 = kronecker_product(I3, kronecker_product(D2 * T2, D1 * T1));
        add_scaled_sparse(triplets, term12, coeff12);

        // Mixed term 13: rho_13 * sigma_1 * sigma_3 * s_1 * s_3 * d2u/ds_1 ds_3
        // = (D3 * T3) \otimes I2 \otimes (D1 * T1)
        SpMat term13 = kronecker_product(D3 * T3, kronecker_product(I2, D1 * T1));
        add_scaled_sparse(triplets, term13, coeff13);

        // Mixed term 23: rho_23 * sigma_2 * sigma_3 * s_2 * s_3 * d2u/ds_2 ds_3
        // = (D3 * T3) \otimes (D2 * T2) \otimes I1
        SpMat term23 = kronecker_product(D3 * T3, kronecker_product(D2 * T2, I1));
        add_scaled_sparse(triplets, term23, coeff23);

        // First derivative terms: r * s_i * du/ds_i
        // r * s_1 * du/ds_1 = I3 \otimes I2 \otimes (D1 * T1)
        SpMat drift1 = kronecker_product(I3, kronecker_product(I2, D1 * T1));
        add_scaled_sparse(triplets, drift1, r);

        // r * s_2 * du/ds_2 = I3 \otimes (D2 * T2) \otimes I1
        SpMat drift2 = kronecker_product(I3, kronecker_product(D2 * T2, I1));
        add_scaled_sparse(triplets, drift2, r);

        // r * s_3 * du/ds_3 = (D3 * T3) \otimes I2 \otimes I1
        SpMat drift3 = kronecker_product(D3 * T3, kronecker_product(I2, I1));
        add_scaled_sparse(triplets, drift3, r);

        // Discount term: -r * u = -r * I
        SpMat I_full = sparse_identity(N);
        add_scaled_sparse(triplets, I_full, -r);

        // Build final matrix
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();
    }

private:
    void add_scaled_sparse(std::vector<Triplet>& triplets, const SpMat& M, double scale) {
        for (int k = 0; k < M.outerSize(); ++k) {
            for (SpMat::InnerIterator it(M, k); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), scale * it.value());
            }
        }
    }
};

/**
 * Split operator into components for ADI methods.
 * A = A0 + A1 + A2 + A3
 * where A0 contains mixed derivatives (treated explicitly)
 * and A1, A2, A3 contain derivatives in each direction (treated implicitly)
 */
struct ADISplit {
    SpMat A0;  // Mixed derivatives
    SpMat A1;  // Direction 1 terms
    SpMat A2;  // Direction 2 terms
    SpMat A3;  // Direction 3 terms

    ADISplit(const Mesh3D& mesh, double r,
             const Eigen::Vector3d& sigma,
             const Eigen::Matrix3d& rho) {

        int N1 = mesh.N1, N2 = mesh.N2, N3 = mesh.N3;
        int N = N1 * N2 * N3;

        // Build 1D operators
        SpMat T1 = build_first_derivative(mesh.mesh1);
        SpMat T2 = build_first_derivative(mesh.mesh2);
        SpMat T3 = build_first_derivative(mesh.mesh3);

        SpMat S1 = build_second_derivative(mesh.mesh1);
        SpMat S2 = build_second_derivative(mesh.mesh2);
        SpMat S3 = build_second_derivative(mesh.mesh3);

        SpMat I1 = sparse_identity(N1);
        SpMat I2 = sparse_identity(N2);
        SpMat I3 = sparse_identity(N3);

        // Diagonal matrices
        SpMat D1(N1, N1), D2(N2, N2), D3(N3, N3);
        {
            std::vector<Triplet> t1, t2, t3;
            for (int i = 0; i < N1; ++i) t1.emplace_back(i, i, mesh.mesh1.points(i));
            for (int i = 0; i < N2; ++i) t2.emplace_back(i, i, mesh.mesh2.points(i));
            for (int i = 0; i < N3; ++i) t3.emplace_back(i, i, mesh.mesh3.points(i));
            D1.setFromTriplets(t1.begin(), t1.end());
            D2.setFromTriplets(t2.begin(), t2.end());
            D3.setFromTriplets(t3.begin(), t3.end());
        }

        SpMat D1_sq = D1 * D1;
        SpMat D2_sq = D2 * D2;
        SpMat D3_sq = D3 * D3;

        double coeff11 = 0.5 * sigma(0) * sigma(0);
        double coeff22 = 0.5 * sigma(1) * sigma(1);
        double coeff33 = 0.5 * sigma(2) * sigma(2);
        double coeff12 = sigma(0) * sigma(1) * rho(0, 1);
        double coeff13 = sigma(0) * sigma(2) * rho(0, 2);
        double coeff23 = sigma(1) * sigma(2) * rho(1, 2);

        // A1: terms in direction 1
        // (1/2) * sigma_1^2 * s_1^2 * d2u/ds_1^2 + r * s_1 * du/ds_1 - (r/3) * u
        A1 = coeff11 * kronecker_product(I3, kronecker_product(I2, D1_sq * S1))
           + r * kronecker_product(I3, kronecker_product(I2, D1 * T1))
           - (r / 3.0) * sparse_identity(N);

        // A2: terms in direction 2
        A2 = coeff22 * kronecker_product(I3, kronecker_product(D2_sq * S2, I1))
           + r * kronecker_product(I3, kronecker_product(D2 * T2, I1))
           - (r / 3.0) * sparse_identity(N);

        // A3: terms in direction 3
        A3 = coeff33 * kronecker_product(D3_sq * S3, kronecker_product(I2, I1))
           + r * kronecker_product(D3 * T3, kronecker_product(I2, I1))
           - (r / 3.0) * sparse_identity(N);

        // A0: mixed derivative terms
        A0 = coeff12 * kronecker_product(I3, kronecker_product(D2 * T2, D1 * T1))
           + coeff13 * kronecker_product(D3 * T3, kronecker_product(I2, D1 * T1))
           + coeff23 * kronecker_product(D3 * T3, kronecker_product(D2 * T2, I1));

        A1.makeCompressed();
        A2.makeCompressed();
        A3.makeCompressed();
        A0.makeCompressed();
    }
};

} // namespace krylov
