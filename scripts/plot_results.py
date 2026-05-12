#!/usr/bin/env python3
"""
Plot benchmark results for Krylov option pricing methods.

Replicates Figure 1 from Niesen & Wright (2011):
- Error vs CPU time (log-log scale)
- Methods distinguished by color:
  - Crank-Nicolson: red
  - Douglas-Rachford: green
  - Hundsdorfer-Verwer: blue
  - Krylov-expmvp: black
  - Al-Mohy-Higham: cyan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Method colors following the paper
METHOD_COLORS = {
    'Crank-Nicolson': 'red',
    'Douglas-Rachford': 'green',
    'Hundsdorfer-Verwer': 'blue',
    'Krylov-expmvp': 'black',
    'Al-Mohy-Higham': 'cyan'
}

METHOD_MARKERS = {
    'Crank-Nicolson': 'o',
    'Douglas-Rachford': 's',
    'Hundsdorfer-Verwer': '^',
    'Krylov-expmvp': 'D',
    'Al-Mohy-Higham': 'v'
}


def load_results(filename: str = 'results/benchmark_results.csv') -> pd.DataFrame:
    """Load benchmark results from CSV."""
    df = pd.read_csv(filename)
    return df


def plot_error_vs_time(df: pd.DataFrame, option_type: str, grid_points: int,
                       ax: plt.Axes, title: str):
    """
    Plot error vs CPU time for a specific option type and grid size.
    """
    subset = df[(df['option_type'] == option_type) &
                (df['grid_points'] == grid_points)]

    for method in METHOD_COLORS.keys():
        method_data = subset[subset['method'] == method]
        if len(method_data) == 0:
            continue

        ax.loglog(method_data['cpu_time'], method_data['error'],
                  color=METHOD_COLORS[method],
                  marker=METHOD_MARKERS[method],
                  linestyle='-',
                  linewidth=1.5,
                  markersize=6,
                  label=method)

    ax.set_xlabel('CPU Time (seconds)', fontsize=10)
    ax.set_ylabel('Error (Euclidean norm)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xlim([1e-2, 1e4])
    ax.set_ylim([1e-9, 1e1])


def create_figure1_replica(df: pd.DataFrame, output_dir: Path):
    """
    Create a figure similar to Figure 1 in Niesen & Wright (2011).

    2x2 grid:
    - Top row: Rainbow option (31 points, 61 points)
    - Bottom row: Basket option (31 points, 61 points)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Get unique grid sizes
    grid_sizes = sorted(df['grid_points'].unique())
    if len(grid_sizes) < 2:
        grid_sizes = [grid_sizes[0], grid_sizes[0]]

    # Rainbow option
    plot_error_vs_time(df, 'rainbow', grid_sizes[0], axes[0, 0],
                       f'Rainbow Option ({grid_sizes[0]}³ grid)')
    plot_error_vs_time(df, 'rainbow', grid_sizes[1], axes[0, 1],
                       f'Rainbow Option ({grid_sizes[1]}³ grid)')

    # Basket option
    plot_error_vs_time(df, 'basket', grid_sizes[0], axes[1, 0],
                       f'Basket Option ({grid_sizes[0]}³ grid)')
    plot_error_vs_time(df, 'basket', grid_sizes[1], axes[1, 1],
                       f'Basket Option ({grid_sizes[1]}³ grid)')

    # Add legend to first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 0.98), fontsize=9)

    plt.suptitle('Error vs CPU Time for European Option Pricing Methods\n'
                 '(Replication of Figure 1, Niesen & Wright 2011)',
                 fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_error_vs_time.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_error_vs_time.pdf', bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'figure1_error_vs_time.png'}")


def create_method_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Create a comparison showing speedup of Krylov over other methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, option_type in enumerate(['rainbow', 'basket']):
        ax = axes[idx]

        # Get results at approximately 1e-4 error level
        subset = df[df['option_type'] == option_type]

        methods = []
        times_at_1e4 = []

        for method in METHOD_COLORS.keys():
            method_data = subset[subset['method'] == method]
            if len(method_data) == 0:
                continue

            # Find the run closest to 1e-4 error
            target_error = 1e-4
            closest_idx = (method_data['error'] - target_error).abs().idxmin()
            closest_row = method_data.loc[closest_idx]

            # Only include if error is reasonable
            if closest_row['error'] < 1e-2:
                methods.append(method)
                times_at_1e4.append(closest_row['cpu_time'])

        if len(methods) > 0:
            colors = [METHOD_COLORS[m] for m in methods]
            bars = ax.bar(range(len(methods)), times_at_1e4, color=colors)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_ylabel('CPU Time (seconds)')
            ax.set_title(f'{option_type.capitalize()} Option\n(Time to achieve ~1e-4 error)')
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved figure to {output_dir / 'method_comparison.png'}")


def print_summary_table(df: pd.DataFrame):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Best performance for each method")
    print("=" * 80)

    for option_type in df['option_type'].unique():
        print(f"\n{option_type.upper()} OPTION:")
        print("-" * 60)

        subset = df[df['option_type'] == option_type]

        for grid in sorted(subset['grid_points'].unique()):
            print(f"\n  Grid: {grid}³ points")
            grid_data = subset[subset['grid_points'] == grid]

            for method in METHOD_COLORS.keys():
                method_data = grid_data[grid_data['method'] == method]
                if len(method_data) == 0:
                    continue

                # Find best error achieved
                best_idx = method_data['error'].idxmin()
                best = method_data.loc[best_idx]

                print(f"    {method:20s}: error={best['error']:.2e}, "
                      f"time={best['cpu_time']:.3f}s, "
                      f"steps={int(best['num_steps'])}")


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    results_dir = project_dir / 'results'

    # Load results
    results_file = results_dir / 'benchmark_results.csv'
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run the C++ benchmark first:")
        print("  cmake -B build && cmake --build build && ./build/option_pricing")
        return

    df = load_results(results_file)
    print(f"Loaded {len(df)} benchmark results")

    # Print summary
    print_summary_table(df)

    # Create plots
    create_figure1_replica(df, results_dir)
    create_method_comparison(df, results_dir)

    print("\n" + "=" * 80)
    print("Plotting complete!")
    print(f"Output files saved to: {results_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
