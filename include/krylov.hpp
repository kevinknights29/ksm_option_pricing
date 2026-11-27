#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include "solvers.hpp"

namespace krylov {

using SpMat = Eigen::SparseMatrix<double>;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

/**
 * Arnoldi iteration for building Krylov subspace.
 *
 * Given matrix A and vector v, builds orthonormal basis V_m
 * and upper Hessenberg matrix H_m such that:
 *   A * V_m = V_m * H_m + h_{m+1,m} * v_{m+1} * e_m^T
 *
 * Following Algorithm 1 in Niesen & Wright (2011).
 */
class ArnoldiIteration {
public:
    Mat V;          // Orthonormal basis (N x m)
    Mat H;          // Upper Hessenberg matrix (m+1 x m)
    double beta;    // Initial norm ||v||
    int m;          // Actual dimension achieved

    void compute(const SpMat& A, const Vec& v, int krylov_dim) {
        int N = v.size();
        m = std::min(krylov_dim, N);

        V.resize(N, m + 1);
        H.resize(m + 1, m);
        H.setZero();

        beta = v.norm();
        if (beta < 1e-14) {
            m = 0;
            return;
        }

        V.col(0) = v / beta;

        for (int j = 0; j < m; ++j) {
            // w = A * v_j
            Vec w = A * V.col(j);

            // Modified Gram-Schmidt orthogonalization
            for (int i = 0; i <= j; ++i) {
                H(i, j) = V.col(i).dot(w);
                w -= H(i, j) * V.col(i);
            }

            H(j + 1, j) = w.norm();

            if (H(j + 1, j) < 1e-14) {
                // Happy breakdown - Krylov subspace is invariant
                m = j + 1;
                break;
            }

            if (j + 1 < m) {
                V.col(j + 1) = w / H(j + 1, j);
            }
        }
    }
};

/**
 * Compute matrix exponential of small dense matrix using Pade approximation.
 * Uses scaling and squaring with Pade approximation order 6.
 * This is simpler and more stable than order 13 for our purposes.
 */
inline Mat compute_matrix_exponential(const Mat& A) {
    int n = A.rows();
    if (n == 0) return Mat();

    // Estimate 1-norm
    double norm_A = A.cwiseAbs().colwise().sum().maxCoeff();

    // Determine scaling factor (ensure norm < 0.5)
    int s = 0;
    if (norm_A > 0.5) {
        s = static_cast<int>(std::ceil(std::log2(2.0 * norm_A)));
    }
    s = std::max(0, s);

    double scale = std::pow(2.0, -s);
    Mat A_scaled = scale * A;

    // Pade(6,6) approximation for exp(A) = p6(A) / q6(A).
    const double c[] = {
        1.0, 1.0 / 2.0, 5.0 / 44.0, 1.0 / 66.0,
        1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0
    };

    Mat I = Mat::Identity(n, n);
    Mat A2 = A_scaled * A_scaled;
    Mat A4 = A2 * A2;
    Mat A6 = A4 * A2;

    // U = A * (c1*I + c3*A2 + c5*A4)
    // V = c0*I + c2*A2 + c4*A4 + c6*A6
    Mat U = A_scaled * (c[1] * I + c[3] * A2 + c[5] * A4);
    Mat V = c[0] * I + c[2] * A2 + c[4] * A4 + c[6] * A6;

    Mat P = V + U;  // Numerator
    Mat Q = V - U;  // Denominator

    // Solve Q * R = P for R = exp(A_scaled)
    Eigen::PartialPivLU<Mat> lu(Q);
    Mat R = lu.solve(P);

    // Squaring phase: exp(A) = exp(A_scaled)^(2^s)
    for (int k = 0; k < s; ++k) {
        R = R * R;
    }

    return R;
}

/**
 * Fixed-dimension Arnoldi exponential integrator.
 *
 * Computes exp(t*A)*v using Krylov subspace projection.
 * Uses the fixed-m, adaptive-h strategy described by Sidje (1998).
 */
class KrylovExponentialSolver {
public:
    int krylov_dim = 30;
    double tol = 1e-7;
    int max_restarts = 100;

    struct Stats {
        int num_steps = 0;
        int num_matvecs = 0;
        int rejected_steps = 0;
    };

    Stats stats;

    /**
     * Compute exp(t*A)*v using Krylov subspace method.
     * Uses adaptive time stepping based on error estimate.
     */
    Vec expmv(const SpMat& A, const Vec& v, double t) {
        stats = Stats{};

        if (t == 0.0) return v;
        if (v.norm() < 1e-14) return v;

        Vec u = v;
        double tau_done = 0.0;
        double h = t;  // Start with full interval
        int m = krylov_dim;

        int max_iter = max_restarts;
        int iter = 0;

        while (tau_done < t - 1e-14 && iter < max_iter) {
            ++iter;
            h = std::min(h, t - tau_done);

            // Build Krylov subspace
            ArnoldiIteration arnoldi;
            arnoldi.compute(A, u, m);
            stats.num_matvecs += arnoldi.m;

            if (arnoldi.m == 0) {
                // Zero vector, nothing to do
                break;
            }

            // Compute exp(h * H_m) * e_1
            Mat H_m = arnoldi.H.topLeftCorner(arnoldi.m, arnoldi.m);
            Mat expH = compute_matrix_exponential(h * H_m);

            Vec e1 = Vec::Zero(arnoldi.m);
            e1(0) = 1.0;
            Vec y = expH * e1;

            // Compute approximation
            Vec u_new = arnoldi.beta * arnoldi.V.leftCols(arnoldi.m) * y;

            // Error estimate using the (m+1,m) element of H
            double err_est = 0.0;
            if (arnoldi.m >= m && arnoldi.m > 0) {
                // Error estimate based on Saad (1992)
                double h_mp1_m = arnoldi.H(arnoldi.m, arnoldi.m - 1);
                if (arnoldi.m > 0) {
                    err_est = std::abs(arnoldi.beta * h_mp1_m * y(arnoldi.m - 1));
                }
            }

            // Accept step if error is small enough or step size is tiny
            if (err_est < tol * u.norm() + 1e-14 || h < 1e-12) {
                u = u_new;
                tau_done += h;
                stats.num_steps++;

                // Potentially increase step size
                if (err_est < 0.1 * tol * u.norm() && tau_done < t) {
                    h = std::min(1.2 * h, t - tau_done);
                }
            } else {
                stats.rejected_steps++;
                // Reduce step size
                double factor = 0.9 * std::pow(tol * u.norm() / (err_est + 1e-14), 1.0 / (m + 1));
                factor = std::max(0.2, std::min(factor, 0.9));
                h *= factor;

                // Don't let step size get too small
                if (h < 1e-12) h = 1e-12;
            }
        }

        if (tau_done < t - 1e-14) {
            throw std::runtime_error(
                "Fixed-dimension Arnoldi failed to reach the requested final time");
        }

        return u;
    }

    /**
     * Solve du/dt = A*u from t=0 to t=T with specified tolerance.
     */
    SolverResult solve(const SpMat& A, const Vec& u0, double T) {
        auto start = std::chrono::steady_clock::now();

        Vec u = expmv(A, u0, T);

        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        return {u, elapsed, stats.num_steps, "Fixed-m Arnoldi"};
    }
};

/**
 * Fixed-step single-precision Arnoldi exponential integrator.
 *
 * Uses a fixed number of substeps with single precision
 * matrix exponential computation for the Krylov projection.
 */
class FixedStepFloatArnoldiSolver {
public:
    int krylov_dim = 30;

    SolverResult solve(const SpMat& A, const Vec& u0, double T, int num_steps) {
        auto start = std::chrono::steady_clock::now();

        double h = T / num_steps;
        Vec u = u0;

        for (int step = 0; step < num_steps; ++step) {
            // Build Krylov subspace
            ArnoldiIteration arnoldi;
            arnoldi.compute(A, u, krylov_dim);

            if (arnoldi.m == 0) continue;

            // Compute exp(h * H_m) with single precision
            Mat H_m = arnoldi.H.topLeftCorner(arnoldi.m, arnoldi.m);
            Eigen::MatrixXf H_float = (h * H_m).cast<float>();
            Eigen::MatrixXf expH_float = compute_matrix_exponential_float(H_float);
            Mat expH = expH_float.cast<double>();

            Vec e1 = Vec::Zero(arnoldi.m);
            e1(0) = 1.0;
            Vec y = expH * e1;

            u = arnoldi.beta * arnoldi.V.leftCols(arnoldi.m) * y;
        }

        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        return {u, elapsed, num_steps, "Float Arnoldi"};
    }

private:
    Eigen::MatrixXf compute_matrix_exponential_float(const Eigen::MatrixXf& A) {
        int n = A.rows();
        if (n == 0) return Eigen::MatrixXf();

        float norm_A = A.cwiseAbs().colwise().sum().maxCoeff();

        int s = 0;
        if (norm_A > 0.5f) {
            s = static_cast<int>(std::ceil(std::log2(2.0f * norm_A)));
        }
        s = std::max(0, s);

        float scale = std::pow(2.0f, -s);
        Eigen::MatrixXf A_scaled = scale * A;

        // Pade(6,6) coefficients.
        const float c[] = {
            1.0f, 1.0f / 2.0f, 5.0f / 44.0f, 1.0f / 66.0f,
            1.0f / 792.0f, 1.0f / 15840.0f, 1.0f / 665280.0f
        };

        Eigen::MatrixXf I = Eigen::MatrixXf::Identity(n, n);
        Eigen::MatrixXf A2 = A_scaled * A_scaled;
        Eigen::MatrixXf A4 = A2 * A2;
        Eigen::MatrixXf A6 = A4 * A2;

        Eigen::MatrixXf U = A_scaled * (c[1] * I + c[3] * A2 + c[5] * A4);
        Eigen::MatrixXf V = c[0] * I + c[2] * A2 + c[4] * A4 + c[6] * A6;

        Eigen::MatrixXf P = V + U;
        Eigen::MatrixXf Q = V - U;

        Eigen::PartialPivLU<Eigen::MatrixXf> lu(Q);
        Eigen::MatrixXf R = lu.solve(P);

        for (int k = 0; k < s; ++k) {
            R = R * R;
        }

        return R;
    }
};

} // namespace krylov
