#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

#include "mesh.hpp"
#include "operators.hpp"
#include "payoffs.hpp"
#include "solvers.hpp"
#include "krylov.hpp"

using namespace krylov;

/**
 * Benchmark result for a single run.
 */
struct BenchmarkResult {
    std::string method;
    std::string option_type;
    int grid_points;
    int num_steps;
    double tolerance;
    double cpu_time;
    double error;
    double spot_price;
};

/**
 * Compute the "exact" solution using a very accurate method.
 * Uses Hundsdorfer-Verwer with many steps as reference.
 */
Eigen::VectorXd compute_exact_solution(const ADISplit& split, const Eigen::VectorXd& u0, double T) {
    // Use HV with many steps as "exact" reference
    HundsdorferVerwerSolver exact_solver;
    int high_steps = 2048;
    auto result = exact_solver.solve(split, u0, T, high_steps);
    std::cout << "  Reference solution computed with HV (" << high_steps << " steps) in "
              << result.cpu_time << " seconds" << std::endl;
    return result.solution;
}

/**
 * Run all benchmarks for a given grid size and option type.
 */
std::vector<BenchmarkResult> run_benchmarks(int N_points,
                                             const std::string& option_type,
                                             const OptionParameters& params) {
    std::vector<BenchmarkResult> results;

    std::cout << "\n=== Running benchmarks: " << option_type
              << " option, " << N_points << " grid points ===" << std::endl;

    // Create mesh
    double c = params.K / 5.0;  // Concentration parameter
    Mesh1D mesh1(N_points, 0.0, params.S_max, params.K, c);
    Mesh1D mesh2(N_points, 0.0, params.S_max, params.K, c);
    Mesh1D mesh3(N_points, 0.0, params.S_max, params.K, c);
    Mesh3D mesh(mesh1, mesh2, mesh3);

    std::cout << "  Total grid points: " << mesh.total_points << std::endl;

    // Build PDE operator
    std::cout << "  Building PDE operator..." << std::endl;
    OptionPDEOperator pde_op(mesh, params.r, params.sigma, params.rho);
    std::cout << "  Matrix size: " << pde_op.N << " x " << pde_op.N << std::endl;
    std::cout << "  Non-zeros: " << pde_op.A.nonZeros() << std::endl;

    // Build ADI split
    std::cout << "  Building ADI split..." << std::endl;
    ADISplit adi_split(mesh, params.r, params.sigma, params.rho);

    // Initial condition (payoff)
    Eigen::VectorXd u0;
    double ref_price;
    if (option_type == "rainbow") {
        u0 = rainbow_call_payoff(mesh, params.K);
        ref_price = params.rainbow_ref;
    } else {
        u0 = basket_call_payoff(mesh, params.K, params.weights);
        ref_price = params.basket_ref;
    }

    // Compute "exact" solution using HV with many steps
    std::cout << "  Computing reference solution..." << std::endl;
    Eigen::VectorXd u_exact = compute_exact_solution(adi_split, u0, params.T);

    int spot_idx = find_spot_index(mesh, params.K);
    double exact_spot = u_exact(spot_idx);
    std::cout << "  Reference spot price: " << exact_spot
              << " (analytical ref: " << ref_price << ")" << std::endl;

    // Step counts for fixed-step methods
    std::vector<int> step_counts = {16, 32, 64, 128, 256, 512};

    // Tolerances for adaptive methods
    std::vector<double> tolerances = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7};

    // 1. Crank-Nicolson (skip for large grids - too slow)
    if (N_points <= 31) {
        std::cout << "\n  Running Crank-Nicolson..." << std::endl;
        CrankNicolsonSolver cn_solver;
        for (int steps : step_counts) {
            auto result = cn_solver.solve(pde_op.A, u0, params.T, steps);
            double error = compute_cube_error(result.solution, u_exact, mesh, params.K);
            results.push_back({
                "Crank-Nicolson", option_type, N_points, steps, 0.0,
                result.cpu_time, error, result.solution(spot_idx)
            });
            std::cout << "    Steps=" << steps << ", Time=" << std::fixed << std::setprecision(3)
                      << result.cpu_time << "s, Error=" << std::scientific << error << std::endl;
        }
    }

    // 2. Douglas-Rachford ADI
    std::cout << "\n  Running Douglas-Rachford..." << std::endl;
    DouglasRachfordSolver dr_solver;
    for (int steps : step_counts) {
        auto result = dr_solver.solve(adi_split, u0, params.T, steps);
        double error = compute_cube_error(result.solution, u_exact, mesh, params.K);
        results.push_back({
            "Douglas-Rachford", option_type, N_points, steps, 0.0,
            result.cpu_time, error, result.solution(spot_idx)
        });
        std::cout << "    Steps=" << steps << ", Time=" << std::fixed << std::setprecision(3)
                  << result.cpu_time << "s, Error=" << std::scientific << error << std::endl;
    }

    // 3. Hundsdorfer-Verwer ADI
    std::cout << "\n  Running Hundsdorfer-Verwer..." << std::endl;
    HundsdorferVerwerSolver hv_solver;
    for (int steps : step_counts) {
        auto result = hv_solver.solve(adi_split, u0, params.T, steps);
        double error = compute_cube_error(result.solution, u_exact, mesh, params.K);
        results.push_back({
            "Hundsdorfer-Verwer", option_type, N_points, steps, 0.0,
            result.cpu_time, error, result.solution(spot_idx)
        });
        std::cout << "    Steps=" << steps << ", Time=" << std::fixed << std::setprecision(3)
                  << result.cpu_time << "s, Error=" << std::scientific << error << std::endl;
    }

    // 4. Krylov exponential integrator
    std::cout << "\n  Running Krylov-expmvp..." << std::endl;
    for (double tol : tolerances) {
        KrylovExponentialSolver krylov_solver;
        krylov_solver.tol = tol;
        krylov_solver.krylov_dim = 30;
        auto result = krylov_solver.solve(pde_op.A, u0, params.T);
        double error = compute_cube_error(result.solution, u_exact, mesh, params.K);
        results.push_back({
            "Krylov-expmvp", option_type, N_points, result.num_steps, tol,
            result.cpu_time, error, result.solution(spot_idx)
        });
        std::cout << "    Tol=" << std::scientific << tol
                  << ", Time=" << std::fixed << std::setprecision(3) << result.cpu_time
                  << "s, Error=" << std::scientific << error
                  << ", Steps=" << result.num_steps << std::endl;
    }

    // 5. Al-Mohy-Higham single precision
    std::cout << "\n  Running Al-Mohy-Higham..." << std::endl;
    AlMohyHighamSolver amh_solver;
    std::vector<int> amh_steps = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    for (int steps : amh_steps) {
        auto result = amh_solver.solve(pde_op.A, u0, params.T, steps);
        double error = compute_cube_error(result.solution, u_exact, mesh, params.K);
        results.push_back({
            "Al-Mohy-Higham", option_type, N_points, steps, 0.0,
            result.cpu_time, error, result.solution(spot_idx)
        });
        std::cout << "    Steps=" << steps << ", Time=" << std::fixed << std::setprecision(3)
                  << result.cpu_time << "s, Error=" << std::scientific << error << std::endl;
    }

    return results;
}

/**
 * Write results to CSV file.
 */
void write_results_csv(const std::vector<BenchmarkResult>& results,
                       const std::string& filename) {
    std::ofstream file(filename);
    file << "method,option_type,grid_points,num_steps,tolerance,cpu_time,error,spot_price\n";

    for (const auto& r : results) {
        file << r.method << ","
             << r.option_type << ","
             << r.grid_points << ","
             << r.num_steps << ","
             << std::scientific << r.tolerance << ","
             << r.cpu_time << ","
             << r.error << ","
             << std::fixed << std::setprecision(6) << r.spot_price << "\n";
    }

    file.close();
    std::cout << "\nResults written to " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "Krylov Subspace Methods for European Option Pricing" << std::endl;
    std::cout << "Replicating Section 5 of Niesen & Wright (2011)" << std::endl;
    std::cout << "==================================================" << std::endl;

    OptionParameters params;

    std::cout << "\nOption Parameters:" << std::endl;
    std::cout << "  Strike K = " << params.K << std::endl;
    std::cout << "  Risk-free rate r = " << params.r << std::endl;
    std::cout << "  Time to expiry T = " << params.T << std::endl;
    std::cout << "  Volatilities sigma = [" << params.sigma.transpose() << "]" << std::endl;
    std::cout << "  Domain = [0, " << params.S_max << "]^3" << std::endl;
    std::cout << "  Reference Rainbow price = " << params.rainbow_ref << std::endl;
    std::cout << "  Reference Basket price = " << params.basket_ref << std::endl;

    std::vector<BenchmarkResult> all_results;

    // Default grid sizes as in paper: 31 and 61 points per direction
    std::vector<int> grid_sizes = {31, 61};

    // Parse command line for smaller test runs
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--small") {
            grid_sizes = {11, 21};
            std::cout << "\nRunning with smaller grid sizes for testing..." << std::endl;
        } else if (arg == "--medium") {
            grid_sizes = {21, 31};
            std::cout << "\nRunning with medium grid sizes..." << std::endl;
        } else if (arg == "--tiny") {
            grid_sizes = {11};
            std::cout << "\nRunning with tiny grid size for quick test..." << std::endl;
        }
    }

    // Run benchmarks
    for (int N : grid_sizes) {
        auto rainbow_results = run_benchmarks(N, "rainbow", params);
        all_results.insert(all_results.end(), rainbow_results.begin(), rainbow_results.end());

        auto basket_results = run_benchmarks(N, "basket", params);
        all_results.insert(all_results.end(), basket_results.begin(), basket_results.end());
    }

    // Write results
    write_results_csv(all_results, "results/benchmark_results.csv");

    // Print summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total benchmark runs: " << all_results.size() << std::endl;

    // Find best method for each option type at ~1e-4 accuracy
    for (const std::string& opt_type : {"rainbow", "basket"}) {
        for (int N : grid_sizes) {
            std::cout << "\nBest methods for " << opt_type << " (" << N << "^3 grid) at error < 1e-3:" << std::endl;

            std::string best_method;
            double best_time = 1e10;

            for (const auto& r : all_results) {
                if (r.option_type == opt_type && r.grid_points == N) {
                    if (r.error < 1e-3 && r.cpu_time < best_time) {
                        best_time = r.cpu_time;
                        best_method = r.method;
                    }
                }
            }

            if (!best_method.empty()) {
                std::cout << "  " << best_method << " achieves target in "
                          << std::fixed << std::setprecision(3) << best_time << " seconds" << std::endl;
            } else {
                std::cout << "  No method achieved error < 1e-3" << std::endl;
            }
        }
    }

    std::cout << "\nBenchmarking complete!" << std::endl;
    std::cout << "Run 'python3 scripts/plot_results.py' to generate plots." << std::endl;

    return 0;
}
