#include <cmath>
#include <iostream>
#include <stdexcept>

#include "krylov.hpp"
#include "payoffs.hpp"

namespace {

/** Reports a failed scalar comparison. */
bool check_close(const char* name, double actual, double expected, double tolerance) {
    if (std::abs(actual - expected) <= tolerance) {
        return true;
    }
    std::cerr << name << ": expected " << expected << ", got " << actual << '\n';
    return false;
}

/** Checks the dense Pade exponential against an analytical scalar result. */
bool matrix_exponential_is_accurate() {
    krylov::Mat argument(1, 1);
    argument(0, 0) = 0.5;
    const double actual = krylov::compute_matrix_exponential(argument)(0, 0);
    return check_close("matrix exponential", actual, std::exp(0.5), 2e-14);
}

/** Checks exact trilinear interpolation of an affine function. */
bool spot_interpolation_is_exact() {
    const krylov::Mesh1D axis(5, 0.0, 4.0);
    const krylov::Mesh3D mesh(axis, axis, axis);
    krylov::Vec values(mesh.total_points);
    for (int index = 0; index < mesh.total_points; ++index) {
        const auto [i, j, k] = mesh.grid_index(index);
        const auto [x, y, z] = mesh.point(i, j, k);
        values(index) = x + 2.0 * y + 3.0 * z;
    }
    return check_close(
        "spot interpolation", krylov::interpolate_at_spot(values, mesh, 1.25),
        7.5, 1e-14);
}

/** Checks that the cube norm uses exactly 9^3 samples. */
bool cube_error_has_fixed_extent() {
    const krylov::Mesh1D axis(11, 0.0, 10.0);
    const krylov::Mesh3D mesh(axis, axis, axis);
    const krylov::Vec ones = krylov::Vec::Ones(mesh.total_points);
    const krylov::Vec zeros = krylov::Vec::Zero(mesh.total_points);
    const bool norm_is_correct = check_close(
        "cube error", krylov::compute_cube_error(ones, zeros, mesh, 5.0),
        27.0, 1e-14);

    try {
        static_cast<void>(krylov::compute_cube_error(ones, zeros, mesh, 0.0));
    } catch (const std::invalid_argument&) {
        return norm_is_correct;
    }
    std::cerr << "cube error accepted a truncated boundary cube\n";
    return false;
}

/** Checks that an exhausted Arnoldi restart budget is reported as failure. */
bool incomplete_integration_throws() {
    const krylov::SpMat matrix = -krylov::sparse_identity(2);
    const krylov::Vec initial = krylov::Vec::Ones(2);
    krylov::KrylovExponentialSolver solver;
    solver.max_restarts = 0;
    try {
        static_cast<void>(solver.expmv(matrix, initial, 1.0));
    } catch (const std::runtime_error&) {
        return true;
    }
    std::cerr << "Arnoldi returned without reaching the final time\n";
    return false;
}

}

/** Runs focused numerical utility regression checks. */
int main() {
    const bool exponential_passed = matrix_exponential_is_accurate();
    const bool interpolation_passed = spot_interpolation_is_exact();
    const bool cube_passed = cube_error_has_fixed_extent();
    const bool completion_passed = incomplete_integration_throws();
    return exponential_passed && interpolation_passed && cube_passed
        && completion_passed ? 0 : 1;
}
