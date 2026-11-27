#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
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

/** Returns the index of the mesh point closest to value. */
inline int closest_index(const Mesh1D& mesh, double value) {
    const auto begin = mesh.points.data();
    const auto end = begin + mesh.N;
    const auto upper = std::lower_bound(begin, end, value);
    if (upper == begin) return 0;
    if (upper == end) return mesh.N - 1;

    const int upper_index = static_cast<int>(upper - begin);
    const int lower_index = upper_index - 1;
    return value - mesh.points(lower_index) <= mesh.points(upper_index) - value
        ? lower_index : upper_index;
}

/** Find the grid point closest to (spot, spot, spot). */
inline int find_spot_index(const Mesh3D& mesh, double spot) {
    return mesh.linear_index(
        closest_index(mesh.mesh1, spot),
        closest_index(mesh.mesh2, spot),
        closest_index(mesh.mesh3, spot));
}

/** Returns the lower index and upper interpolation weight for value. */
inline std::pair<int, double> interpolation_bracket(const Mesh1D& mesh,
                                                    double value) {
    if (mesh.N < 2 || value < mesh.points(0) || value > mesh.points(mesh.N - 1)) {
        throw std::invalid_argument("Interpolation point is outside the mesh");
    }

    const auto begin = mesh.points.data();
    const auto end = begin + mesh.N;
    const auto upper = std::lower_bound(begin, end, value);
    if (upper == begin) return {0, 0.0};
    if (upper == end) return {mesh.N - 2, 1.0};

    const int lower_index = static_cast<int>(upper - begin) - 1;
    const double lower = mesh.points(lower_index);
    const double weight = (value - lower) /
        (mesh.points(lower_index + 1) - lower);
    return {lower_index, weight};
}

/** Evaluates a grid solution at (spot, spot, spot) by trilinear interpolation. */
inline double interpolate_at_spot(const Eigen::VectorXd& u,
                                  const Mesh3D& mesh,
                                  double spot) {
    if (u.size() != mesh.total_points) {
        throw std::invalid_argument("Solution size does not match the mesh");
    }

    const auto [i0, wi] = interpolation_bracket(mesh.mesh1, spot);
    const auto [j0, wj] = interpolation_bracket(mesh.mesh2, spot);
    const auto [k0, wk] = interpolation_bracket(mesh.mesh3, spot);
    const std::array<double, 2> weights_i = {1.0 - wi, wi};
    const std::array<double, 2> weights_j = {1.0 - wj, wj};
    const std::array<double, 2> weights_k = {1.0 - wk, wk};

    double value = 0.0;
    for (int di = 0; di < 2; ++di) {
        for (int dj = 0; dj < 2; ++dj) {
            for (int dk = 0; dk < 2; ++dk) {
                value += weights_i[di] * weights_j[dj] * weights_k[dk]
                    * u(mesh.linear_index(i0 + di, j0 + dj, k0 + dk));
            }
        }
    }
    return value;
}

/**
 * Compute error on 9x9x9 cube around spot price.
 * Following the paper's methodology.
 */
inline double compute_cube_error(const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& u_exact,
                                  const Mesh3D& mesh,
                                  double spot) {
    if (u.size() != mesh.total_points || u_exact.size() != mesh.total_points) {
        throw std::invalid_argument("Solution size does not match the mesh");
    }

    const int center1 = closest_index(mesh.mesh1, spot);
    const int center2 = closest_index(mesh.mesh2, spot);
    const int center3 = closest_index(mesh.mesh3, spot);
    if (center1 < 4 || center1 + 4 >= mesh.N1 ||
        center2 < 4 || center2 + 4 >= mesh.N2 ||
        center3 < 4 || center3 + 4 >= mesh.N3) {
        throw std::invalid_argument("The requested 9x9x9 error cube does not fit the mesh");
    }

    // Compute error on 9x9x9 cube
    double error_sq = 0.0;
    for (int di = -4; di <= 4; ++di) {
        for (int dj = -4; dj <= 4; ++dj) {
            for (int dk = -4; dk <= 4; ++dk) {
                int i = center1 + di;
                int j = center2 + dj;
                int k = center3 + dk;

                int idx = mesh.linear_index(i, j, k);
                double diff = u(idx) - u_exact(idx);
                error_sq += diff * diff;
            }
        }
    }

    return std::sqrt(error_sq);
}

} // namespace krylov
