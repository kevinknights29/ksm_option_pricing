#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "mesh.hpp"

namespace krylov {

/**
 * Option payoff functions.
 * These create the initial condition u0 for the PDE.
 */

/**
 * Rainbow call option (call on minimum).
 * Payoff = max(min(s1, s2, s3) - K, 0)
 */
inline Eigen::VectorXd rainbow_call_payoff(const Mesh3D& mesh, double K) {
    int N = mesh.total_points;
    Eigen::VectorXd u0(N);

    for (int idx = 0; idx < N; ++idx) {
        auto [i, j, k] = mesh.grid_index(idx);
        auto [s1, s2, s3] = mesh.point(i, j, k);

        double min_s = std::min({s1, s2, s3});
        u0(idx) = std::max(min_s - K, 0.0);
    }

    return u0;
}

/**
 * Basket call option.
 * Payoff = max(sum(w_i * s_i) - K, 0)
 */
inline Eigen::VectorXd basket_call_payoff(const Mesh3D& mesh, double K,
                                          const Eigen::Vector3d& weights) {
    int N = mesh.total_points;
    Eigen::VectorXd u0(N);

    for (int idx = 0; idx < N; ++idx) {
        auto [i, j, k] = mesh.grid_index(idx);
        auto [s1, s2, s3] = mesh.point(i, j, k);

        double basket = weights(0) * s1 + weights(1) * s2 + weights(2) * s3;
        u0(idx) = std::max(basket - K, 0.0);
    }

    return u0;
}

/**
 * Problem parameters from Section 5 of Niesen & Wright (2011).
 */
struct OptionParameters {
    double K = 100.0;           // Strike price
    double r = 0.04;            // Risk-free rate
    double T = 1.0;             // Time to expiry
    Eigen::Vector3d sigma;      // Volatilities
    Eigen::Matrix3d rho;        // Correlation matrix
    Eigen::Vector3d weights;    // Basket weights
    double S_max = 300.0;       // Upper bound of domain

    OptionParameters() {
        sigma << 0.3, 0.35, 0.4;

        rho << 1.0, 0.5, 0.5,
               0.5, 1.0, 0.5,
               0.5, 0.5, 1.0;

        weights << 1.0/3.0, 1.0/3.0, 1.0/3.0;
    }

    // Reference solutions from paper
    static constexpr double rainbow_ref = 4.4450;
    static constexpr double basket_ref = 13.2449;
};

/**
 * Find the index closest to the spot price.
 */
inline int find_spot_index(const Mesh3D& mesh, double spot) {
    int idx1 = 0, idx2 = 0, idx3 = 0;

    // Find closest index in each dimension
    double min_dist1 = std::abs(mesh.mesh1.points(0) - spot);
    for (int i = 1; i < mesh.N1; ++i) {
        double dist = std::abs(mesh.mesh1.points(i) - spot);
        if (dist < min_dist1) {
            min_dist1 = dist;
            idx1 = i;
        }
    }

    double min_dist2 = std::abs(mesh.mesh2.points(0) - spot);
    for (int i = 1; i < mesh.N2; ++i) {
        double dist = std::abs(mesh.mesh2.points(i) - spot);
        if (dist < min_dist2) {
            min_dist2 = dist;
            idx2 = i;
        }
    }

    double min_dist3 = std::abs(mesh.mesh3.points(0) - spot);
    for (int i = 1; i < mesh.N3; ++i) {
        double dist = std::abs(mesh.mesh3.points(i) - spot);
        if (dist < min_dist3) {
            min_dist3 = dist;
            idx3 = i;
        }
    }

    return mesh.linear_index(idx1, idx2, idx3);
}

/**
 * Compute error on 9x9x9 cube around spot price.
 * Following the paper's methodology.
 */
inline double compute_cube_error(const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& u_exact,
                                  const Mesh3D& mesh,
                                  double spot) {
    int center1 = 0, center2 = 0, center3 = 0;

    // Find center indices
    for (int i = 0; i < mesh.N1; ++i) {
        if (mesh.mesh1.points(i) >= spot) {
            center1 = i;
            break;
        }
    }
    for (int i = 0; i < mesh.N2; ++i) {
        if (mesh.mesh2.points(i) >= spot) {
            center2 = i;
            break;
        }
    }
    for (int i = 0; i < mesh.N3; ++i) {
        if (mesh.mesh3.points(i) >= spot) {
            center3 = i;
            break;
        }
    }

    // Compute error on 9x9x9 cube
    double error_sq = 0.0;
    int count = 0;

    for (int di = -4; di <= 4; ++di) {
        for (int dj = -4; dj <= 4; ++dj) {
            for (int dk = -4; dk <= 4; ++dk) {
                int i = center1 + di;
                int j = center2 + dj;
                int k = center3 + dk;

                if (i >= 0 && i < mesh.N1 &&
                    j >= 0 && j < mesh.N2 &&
                    k >= 0 && k < mesh.N3) {
                    int idx = mesh.linear_index(i, j, k);
                    double diff = u(idx) - u_exact(idx);
                    error_sq += diff * diff;
                    count++;
                }
            }
        }
    }

    return std::sqrt(error_sq);
}

} // namespace krylov
