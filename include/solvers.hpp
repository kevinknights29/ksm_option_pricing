#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <chrono>
#include <iostream>
#include "operators.hpp"

namespace krylov {

using SpMat = Eigen::SparseMatrix<double>;
using Vec = Eigen::VectorXd;

/**
 * Result structure for solver benchmarking.
 */
struct SolverResult {
    Vec solution;
    double wall_time_seconds;
    int num_steps;
    std::string method_name;
};

/**
 * Crank-Nicolson method.
 *
 * u_{n+1} = (I - h/2 * A)^{-1} * (I + h/2 * A) * u_n
 *
 * This is the trapezoidal rule, second-order accurate but not L-stable.
 */
class CrankNicolsonSolver {
public:
    SolverResult solve(const SpMat& A, const Vec& u0, double T, int num_steps) {
        auto start = std::chrono::steady_clock::now();

        int N = A.rows();
        double h = T / num_steps;

        // Build (I - h/2 * A) and (I + h/2 * A)
        SpMat I = sparse_identity(N);
        SpMat L = I - (h / 2.0) * A;  // Left matrix
        SpMat R = I + (h / 2.0) * A;  // Right matrix

        // Factorize L
        Eigen::SparseLU<SpMat> solver;
        solver.analyzePattern(L);
        solver.factorize(L);

        if (solver.info() != Eigen::Success) {
            std::cerr << "CrankNicolson: LU factorization failed!" << std::endl;
        }

        Vec u = u0;
        for (int step = 0; step < num_steps; ++step) {
            Vec rhs = R * u;
            u = solver.solve(rhs);
        }

        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        return {u, elapsed, num_steps, "Crank-Nicolson"};
    }
};

/**
 * Douglas-Rachford ADI method (theta = 1/2).
 *
 * Following Douglas and Rachford (1956).
 * First-order accurate in time.
 *
 * A = A0 + A1 + A2 + A3
 * where A0 is treated explicitly, A1, A2, A3 implicitly.
 */
class DouglasRachfordSolver {
public:
    SolverResult solve(const ADISplit& split, const Vec& u0, double T, int num_steps) {
        auto start = std::chrono::steady_clock::now();

        int N = split.A1.rows();
        double h = T / num_steps;
        double theta = 0.5;

        SpMat I = sparse_identity(N);

        // Build implicit operators (I - theta * h * Ai)
        SpMat L1 = I - theta * h * split.A1;
        SpMat L2 = I - theta * h * split.A2;
        SpMat L3 = I - theta * h * split.A3;

        // Factorize
        Eigen::SparseLU<SpMat> solver1, solver2, solver3;
        solver1.analyzePattern(L1);
        solver1.factorize(L1);
        solver2.analyzePattern(L2);
        solver2.factorize(L2);
        solver3.analyzePattern(L3);
        solver3.factorize(L3);

        // Full operator for explicit part
        SpMat A_full = split.A0 + split.A1 + split.A2 + split.A3;

        Vec u = u0;
        for (int step = 0; step < num_steps; ++step) {
            // Y0 = u_n + h * A * u_n
            Vec Y0 = u + h * (A_full * u);

            // Y1 = Y0 + theta * h * A1 * (Y1 - u_n)
            // => (I - theta * h * A1) * Y1 = Y0 - theta * h * A1 * u_n
            Vec rhs1 = Y0 - theta * h * (split.A1 * u);
            Vec Y1 = solver1.solve(rhs1);

            // Y2 = Y1 + theta * h * A2 * (Y2 - u_n)
            Vec rhs2 = Y1 - theta * h * (split.A2 * u);
            Vec Y2 = solver2.solve(rhs2);

            // Y3 = Y2 + theta * h * A3 * (Y3 - u_n)
            Vec rhs3 = Y2 - theta * h * (split.A3 * u);
            Vec Y3 = solver3.solve(rhs3);

            u = Y3;
        }

        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        return {u, elapsed, num_steps, "Douglas-Rachford"};
    }
};

/**
 * Hundsdorfer-Verwer ADI method (theta = 1/2).
 *
 * Following Hundsdorfer and Verwer (2003).
 * Second-order accurate in time.
 */
class HundsdorferVerwerSolver {
public:
    SolverResult solve(const ADISplit& split, const Vec& u0, double T, int num_steps) {
        auto start = std::chrono::steady_clock::now();

        int N = split.A1.rows();
        double h = T / num_steps;
        double theta = 0.5;

        SpMat I = sparse_identity(N);

        // Build implicit operators
        SpMat L1 = I - theta * h * split.A1;
        SpMat L2 = I - theta * h * split.A2;
        SpMat L3 = I - theta * h * split.A3;

        // Factorize
        Eigen::SparseLU<SpMat> solver1, solver2, solver3;
        solver1.analyzePattern(L1);
        solver1.factorize(L1);
        solver2.analyzePattern(L2);
        solver2.factorize(L2);
        solver3.analyzePattern(L3);
        solver3.factorize(L3);

        SpMat A_full = split.A0 + split.A1 + split.A2 + split.A3;

        Vec u = u0;
        for (int step = 0; step < num_steps; ++step) {
            // Stage 1: Y0 = u_n + h * A * u_n
            Vec Y0 = u + h * (A_full * u);

            // Stage 2-4: Implicit corrections (first sweep)
            Vec rhs1 = Y0 - theta * h * (split.A1 * u);
            Vec Y1 = solver1.solve(rhs1);

            Vec rhs2 = Y1 - theta * h * (split.A2 * u);
            Vec Y2 = solver2.solve(rhs2);

            Vec rhs3 = Y2 - theta * h * (split.A3 * u);
            Vec Y3 = solver3.solve(rhs3);

            /**
             * Stage 5: explicit correction.
             * Y_tilde_0 = Y0 + (1/2) * h * (A * Y3 - A * u_n).
             */
            Vec Y_tilde_0 = Y0 + 0.5 * h * (A_full * Y3 - A_full * u);

            // Stage 6-8: Implicit corrections (second sweep)
            Vec rhs4 = Y_tilde_0 - theta * h * (split.A1 * Y3);
            Vec Y_tilde_1 = solver1.solve(rhs4);

            Vec rhs5 = Y_tilde_1 - theta * h * (split.A2 * Y3);
            Vec Y_tilde_2 = solver2.solve(rhs5);

            Vec rhs6 = Y_tilde_2 - theta * h * (split.A3 * Y3);
            Vec Y_tilde_3 = solver3.solve(rhs6);

            u = Y_tilde_3;
        }

        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        return {u, elapsed, num_steps, "Hundsdorfer-Verwer"};
    }
};

} // namespace krylov
