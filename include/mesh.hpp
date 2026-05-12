#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace krylov {

/**
 * Non-uniform mesh generation using sinh transformation.
 * Following Tavella and Randall (2000) and Kluge (2002).
 *
 * The mesh concentrates points around the strike price K.
 * x_i = K + c * sinh(xi_i)
 * where xi_i are equidistant points.
 */
class Mesh1D {
public:
    Eigen::VectorXd points;    // Grid points
    Eigen::VectorXd dx;        // Grid spacings dx[i] = x[i+1] - x[i]
    int N;                     // Number of points
    double L, U;               // Lower and upper bounds
    double K;                  // Concentration point (strike)
    double c;                  // Concentration parameter

    Mesh1D(int num_points, double lower, double upper, double strike, double concentration)
        : N(num_points), L(lower), U(upper), K(strike), c(concentration) {

        points.resize(N);
        dx.resize(N - 1);

        // Create equidistant points in xi-space
        double xi_L = std::asinh((L - K) / c);
        double xi_U = std::asinh((U - K) / c);
        double d_xi = (xi_U - xi_L) / (N - 1);

        // Transform to x-space using sinh
        for (int i = 0; i < N; ++i) {
            double xi_i = xi_L + i * d_xi;
            points(i) = K + c * std::sinh(xi_i);
        }

        // Compute grid spacings
        for (int i = 0; i < N - 1; ++i) {
            dx(i) = points(i + 1) - points(i);
        }
    }

    // Uniform mesh constructor
    Mesh1D(int num_points, double lower, double upper)
        : N(num_points), L(lower), U(upper), K((lower + upper) / 2), c(0) {

        points.resize(N);
        dx.resize(N - 1);

        double h = (U - L) / (N - 1);
        for (int i = 0; i < N; ++i) {
            points(i) = L + i * h;
        }
        for (int i = 0; i < N - 1; ++i) {
            dx(i) = h;
        }
    }
};

/**
 * 3D mesh for option pricing.
 * Combines three 1D meshes for s1, s2, s3 directions.
 */
class Mesh3D {
public:
    Mesh1D mesh1, mesh2, mesh3;
    int N1, N2, N3;
    int total_points;

    Mesh3D(const Mesh1D& m1, const Mesh1D& m2, const Mesh1D& m3)
        : mesh1(m1), mesh2(m2), mesh3(m3),
          N1(m1.N), N2(m2.N), N3(m3.N),
          total_points(m1.N * m2.N * m3.N) {}

    // Convert 3D index (i, j, k) to linear index
    int linear_index(int i, int j, int k) const {
        return i + j * N1 + k * N1 * N2;
    }

    // Convert linear index to 3D indices
    std::tuple<int, int, int> grid_index(int idx) const {
        int k = idx / (N1 * N2);
        int rem = idx % (N1 * N2);
        int j = rem / N1;
        int i = rem % N1;
        return {i, j, k};
    }

    // Get grid point values
    std::tuple<double, double, double> point(int i, int j, int k) const {
        return {mesh1.points(i), mesh2.points(j), mesh3.points(k)};
    }
};

} // namespace krylov
