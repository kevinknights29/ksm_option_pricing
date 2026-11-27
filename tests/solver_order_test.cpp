#include <cmath>
#include <iostream>
#include <string_view>

#include "solvers.hpp"

namespace {

using krylov::ADISplit;
using krylov::DouglasRachfordSolver;
using krylov::HundsdorferVerwerSolver;
using krylov::Mesh1D;
using krylov::Mesh3D;
using krylov::SpMat;
using krylov::Vec;

/**
 * Builds a block-diagonal test split with noncommuting directional parts.
 *
 * Each 2-by-2 block has a closed-form exponential while A1*A2 differs from
 * A2*A1, exercising the noncommuting ADI recurrence.
 */
ADISplit make_test_split() {
    const Mesh1D mesh_1d(2, 0.0, 1.0);
    const Mesh3D mesh(mesh_1d, mesh_1d, mesh_1d);
    const Eigen::Vector3d sigma = Eigen::Vector3d::Zero();
    const Eigen::Matrix3d rho = Eigen::Matrix3d::Identity();
    ADISplit split(mesh, 0.0, sigma, rho);

    split.A0 = -0.1 * krylov::sparse_identity(mesh.total_points);
    split.A1 = -0.3 * krylov::sparse_identity(mesh.total_points);
    split.A2 = -0.3 * krylov::sparse_identity(mesh.total_points);
    split.A3 = -0.3 * krylov::sparse_identity(mesh.total_points);
    for (int block = 0; block < mesh.total_points; block += 2) {
        split.A1.coeffRef(block, block + 1) = 0.1;
        split.A2.coeffRef(block + 1, block) = 0.2;
    }
    split.A1.makeCompressed();
    split.A2.makeCompressed();
    return split;
}

/** Returns the analytical solution at T = 1 for the block test problem. */
Vec exact_solution(int size) {
    Vec exact(size);
    const double delta = std::sqrt(0.02);
    const double diagonal_factor = std::exp(-1.0) * std::cosh(delta);
    const double off_diagonal_factor = std::exp(-1.0) * std::sinh(delta) / delta;
    for (int block = 0; block < size; block += 2) {
        exact(block) = diagonal_factor + off_diagonal_factor * 0.05;
        exact(block + 1) = 0.5 * diagonal_factor + off_diagonal_factor * 0.2;
    }
    return exact;
}

/** Returns the Euclidean error at T = 1 for a solver and step count. */
template <typename Solver>
double error_for(Solver& solver, const ADISplit& split, int steps) {
    Vec initial(split.A0.rows());
    for (int block = 0; block < initial.size(); block += 2) {
        initial(block) = 1.0;
        initial(block + 1) = 0.5;
    }
    const Vec exact = exact_solution(initial.size());
    return (solver.solve(split, initial, 1.0, steps).solution - exact).norm();
}

/** Returns the observed order when the time step is halved. */
double observed_order(double coarse_error, double fine_error) {
    return std::log2(coarse_error / fine_error);
}

/** Checks an observed convergence order against a closed interval. */
bool check_order(std::string_view method, double order, double lower, double upper) {
    if (order >= lower && order <= upper) {
        return true;
    }

    std::cerr << method << " observed order " << order
              << " is outside [" << lower << ", " << upper << "]\n";
    return false;
}

}

/** Runs temporal order regression checks for both ADI methods. */
int main() {
    const ADISplit split = make_test_split();
    DouglasRachfordSolver douglas;
    HundsdorferVerwerSolver hundsdorfer_verwer;

    const double douglas_order = observed_order(
        error_for(douglas, split, 32), error_for(douglas, split, 64));
    const double hv_order = observed_order(
        error_for(hundsdorfer_verwer, split, 16),
        error_for(hundsdorfer_verwer, split, 32));

    std::cout << "Douglas-Rachford observed order: " << douglas_order << '\n'
              << "Hundsdorfer-Verwer observed order: " << hv_order << '\n';

    const bool douglas_passed = check_order(
        "Douglas-Rachford", douglas_order, 0.8, 1.2);
    const bool hv_passed = check_order(
        "Hundsdorfer-Verwer", hv_order, 1.8, 2.2);
    return douglas_passed && hv_passed ? 0 : 1;
}
