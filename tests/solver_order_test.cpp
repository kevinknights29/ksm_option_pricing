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
 * Builds a diagonal test split for the scalar equation u' = -u.
 *
 * The eight identical degrees of freedom exercise every ADI stage while
 * retaining the analytical solution u(t) = exp(-t) u(0).
 */
ADISplit make_test_split() {
    const Mesh1D mesh_1d(2, 0.0, 1.0);
    const Mesh3D mesh(mesh_1d, mesh_1d, mesh_1d);
    const Eigen::Vector3d sigma = Eigen::Vector3d::Zero();
    const Eigen::Matrix3d rho = Eigen::Matrix3d::Identity();
    ADISplit split(mesh, 0.0, sigma, rho);

    split.A0 = -0.1 * krylov::sparse_identity(mesh.total_points);
    split.A1 = -0.2 * krylov::sparse_identity(mesh.total_points);
    split.A2 = -0.3 * krylov::sparse_identity(mesh.total_points);
    split.A3 = -0.4 * krylov::sparse_identity(mesh.total_points);
    return split;
}

/** Returns the Euclidean error at T = 1 for a solver and step count. */
template <typename Solver>
double error_for(Solver& solver, const ADISplit& split, int steps) {
    const Vec initial = Vec::Ones(split.A0.rows());
    const Vec exact = std::exp(-1.0) * initial;
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
