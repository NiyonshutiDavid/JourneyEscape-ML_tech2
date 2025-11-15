#!/usr/bin/env python3
"""
DQN Experiment Results Analysis Script
=====================================

This script analyzes the results from hyperparameter experiments and generates
comprehensive reports and visualizations to understand which configurations
work best for the JourneyEscape environment.

Usage:
    python analyze_results.py
    python analyze_results.py --member_name "John_Doe" --policy CNN
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ExperimentAnalyzer:
    """Analyzes DQN hyperparameter experiment results."""

    def __init__(self, results_file: str = "experiments/hyperparameter_results.csv"):
        self.results_file = results_file
        self.df = None
        self.load_data()

    def load_data(self):
        """Load experiment results from CSV file."""
        if not os.path.exists(self.results_file):
            print(f"❌ Results file not found: {self.results_file}")
            print("   Run some experiments first with train.py or run_experiments.py")
            return

        try:
            self.df = pd.read_csv(self.results_file)
            print(
                f"✅ Loaded {len(self.df)} experiment results from {self.results_file}"
            )

            # Clean and prepare data
            self._clean_data()

        except Exception as e:
            print(f"❌ Error loading results: {e}")
            self.df = None

    def _clean_data(self):
        """Clean and prepare the data for analysis."""
        if self.df is None:
            return

        # Convert timestamp to datetime
        if "timestamp" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Ensure numeric columns are properly typed
        numeric_cols = [
            "learning_rate",
            "gamma",
            "batch_size",
            "epsilon_start",
            "epsilon_end",
            "avg_reward",
            "std_reward",
            "min_reward",
            "max_reward",
            "avg_episode_length",
            "training_time_seconds",
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        print(f"   Data shape: {self.df.shape}")
        print(f"   Columns: {list(self.df.columns)}")

    def generate_summary_report(
        self, member_name: Optional[str] = None, policy: Optional[str] = None
    ) -> str:
        """Generate a comprehensive summary report."""
        if self.df is None or len(self.df) == 0:
            return "No data available for analysis."

        # Filter data if requested
        filtered_df = self.df.copy()
        if member_name:
            filtered_df = filtered_df[
                filtered_df["experiment_name"].str.contains(member_name, na=False)
            ]
        if policy:
            filtered_df = filtered_df[filtered_df["policy_type"] == policy]

        if len(filtered_df) == 0:
            return f"No experiments found for member: {member_name}, policy: {policy}"

        report = []
        report.append("=" * 80)
        report.append("🧪 DQN HYPERPARAMETER EXPERIMENT ANALYSIS REPORT")
        report.append("=" * 80)

        # Basic statistics
        report.append(f"\n📊 DATASET OVERVIEW")
        report.append(f"Total experiments analyzed: {len(filtered_df)}")
        if member_name:
            report.append(f"Member: {member_name}")
        if policy:
            report.append(f"Policy: {policy}")

        # Performance statistics
        if "avg_reward" in filtered_df.columns:
            report.append(f"\n🎯 PERFORMANCE STATISTICS")
            report.append(f"Average Reward:")
            report.append(f"  Best:    {filtered_df['avg_reward'].max():.2f}")
            report.append(f"  Worst:   {filtered_df['avg_reward'].min():.2f}")
            report.append(f"  Mean:    {filtered_df['avg_reward'].mean():.2f}")
            report.append(f"  Std:     {filtered_df['avg_reward'].std():.2f}")

        # Best configurations
        if "avg_reward" in filtered_df.columns:
            report.append(f"\n🏆 TOP 3 CONFIGURATIONS (by average reward)")
            top_configs = filtered_df.nlargest(3, "avg_reward")

            for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                report.append(f"\n{i}. Avg Reward: {row['avg_reward']:.2f}")
                if "hyperparameter_set_name" in row:
                    report.append(f"   Configuration: {row['hyperparameter_set_name']}")
                report.append(f"   Learning Rate: {row.get('learning_rate', 'N/A')}")
                report.append(f"   Gamma: {row.get('gamma', 'N/A')}")
                report.append(f"   Batch Size: {row.get('batch_size', 'N/A')}")
                report.append(f"   Epsilon Start: {row.get('epsilon_start', 'N/A')}")
                report.append(f"   Epsilon End: {row.get('epsilon_end', 'N/A')}")

        # Hyperparameter impact analysis
        report.append(f"\n🔬 HYPERPARAMETER IMPACT ANALYSIS")

        # Learning rate analysis
        if (
            "learning_rate" in filtered_df.columns
            and "avg_reward" in filtered_df.columns
        ):
            lr_groups = filtered_df.groupby("learning_rate")["avg_reward"].agg(
                ["mean", "std", "count"]
            )
            report.append(f"\nLearning Rate Impact:")
            for lr, stats in lr_groups.iterrows():
                report.append(
                    f"  LR {lr:.0e}: Avg={stats['mean']:.2f}, Std={stats['std']:.2f}, Count={stats['count']}"
                )

        # Gamma analysis
        if "gamma" in filtered_df.columns and "avg_reward" in filtered_df.columns:
            gamma_groups = filtered_df.groupby("gamma")["avg_reward"].agg(
                ["mean", "std", "count"]
            )
            report.append(f"\nGamma (Discount Factor) Impact:")
            for gamma, stats in gamma_groups.iterrows():
                report.append(
                    f"  Gamma {gamma:.3f}: Avg={stats['mean']:.2f}, Std={stats['std']:.2f}, Count={stats['count']}"
                )

        # Batch size analysis
        if "batch_size" in filtered_df.columns and "avg_reward" in filtered_df.columns:
            batch_groups = filtered_df.groupby("batch_size")["avg_reward"].agg(
                ["mean", "std", "count"]
            )
            report.append(f"\nBatch Size Impact:")
            for batch_size, stats in batch_groups.iterrows():
                report.append(
                    f"  Batch {int(batch_size)}: Avg={stats['mean']:.2f}, Std={stats['std']:.2f}, Count={stats['count']}"
                )

        # Training efficiency
        if "training_time_seconds" in filtered_df.columns:
            report.append(f"\n⏱️  TRAINING EFFICIENCY")
            report.append(f"Training Time:")
            report.append(
                f"  Fastest: {filtered_df['training_time_seconds'].min() / 3600:.2f} hours"
            )
            report.append(
                f"  Slowest: {filtered_df['training_time_seconds'].max() / 3600:.2f} hours"
            )
            report.append(
                f"  Average: {filtered_df['training_time_seconds'].mean() / 3600:.2f} hours"
            )

        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS")

        if "avg_reward" in filtered_df.columns:
            best_exp = filtered_df.loc[filtered_df["avg_reward"].idxmax()]
            report.append(f"1. Best overall configuration:")
            report.append(f"   - Learning Rate: {best_exp.get('learning_rate', 'N/A')}")
            report.append(f"   - Gamma: {best_exp.get('gamma', 'N/A')}")
            report.append(f"   - Batch Size: {best_exp.get('batch_size', 'N/A')}")
            report.append(
                f"   - This achieved {best_exp['avg_reward']:.2f} average reward"
            )

        # Policy comparison if both available
        if (
            "policy_type" in filtered_df.columns
            and filtered_df["policy_type"].nunique() > 1
        ):
            policy_comparison = filtered_df.groupby("policy_type")["avg_reward"].agg(
                ["mean", "std", "count"]
            )
            report.append(f"\n2. Policy Comparison:")
            for policy_name, stats in policy_comparison.iterrows():
                report.append(
                    f"   - {policy_name}: Avg={stats['mean']:.2f}, Std={stats['std']:.2f}, Count={stats['count']}"
                )

        report.append(f"\n" + "=" * 80)

        return "\n".join(report)

    def create_visualizations(self, output_dir: str = "experiments/plots"):
        """Create comprehensive visualizations of experiment results."""
        if self.df is None or len(self.df) == 0:
            print("❌ No data available for visualization")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"📊 Creating visualizations in {output_dir}/")

        # Set up the plotting style
        plt.rcParams["figure.figsize"] = (12, 8)

        # 1. Hyperparameter vs Performance plots
        self._plot_hyperparameter_impact(output_dir)

        # 2. Performance distribution
        self._plot_performance_distribution(output_dir)

        # 3. Training time analysis
        self._plot_training_efficiency(output_dir)

        # 4. Correlation heatmap
        self._plot_correlation_heatmap(output_dir)

        # 5. Policy comparison if available
        self._plot_policy_comparison(output_dir)

        print(f"✅ Visualizations saved to {output_dir}/")

    def _plot_hyperparameter_impact(self, output_dir: str):
        """Plot the impact of different hyperparameters on performance."""
        if "avg_reward" not in self.df.columns:
            return

        hyperparams = [
            "learning_rate",
            "gamma",
            "batch_size",
            "epsilon_start",
            "epsilon_end",
        ]
        available_params = [p for p in hyperparams if p in self.df.columns]

        if not available_params:
            return

        n_params = len(available_params)
        n_cols = 2
        n_rows = (n_params + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, param in enumerate(available_params):
            ax = axes[i]

            # Create boxplot for categorical or few unique values
            unique_vals = self.df[param].nunique()
            if unique_vals <= 10:
                sns.boxplot(data=self.df, x=param, y="avg_reward", ax=ax)
                ax.set_title(f"Average Reward by {param}")
                plt.setp(ax.get_xticklabels(), rotation=45)
            else:
                # Scatter plot for continuous variables
                ax.scatter(self.df[param], self.df["avg_reward"], alpha=0.6)
                ax.set_xlabel(param)
                ax.set_ylabel("Average Reward")
                ax.set_title(f"Average Reward vs {param}")

        # Hide unused subplots
        for i in range(len(available_params), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "hyperparameter_impact.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_performance_distribution(self, output_dir: str):
        """Plot the distribution of performance across experiments."""
        if "avg_reward" not in self.df.columns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        axes[0].hist(self.df["avg_reward"], bins=20, alpha=0.7, edgecolor="black")
        axes[0].set_xlabel("Average Reward")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Distribution of Average Rewards")
        axes[0].axvline(
            self.df["avg_reward"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {self.df['avg_reward'].mean():.2f}",
        )
        axes[0].legend()

        # Box plot with individual points
        axes[1].boxplot(self.df["avg_reward"])
        axes[1].scatter(
            [1] * len(self.df), self.df["avg_reward"], alpha=0.5, color="red"
        )
        axes[1].set_ylabel("Average Reward")
        axes[1].set_title("Performance Distribution")
        axes[1].set_xticklabels(["All Experiments"])

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "performance_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_training_efficiency(self, output_dir: str):
        """Plot training time vs performance."""
        if (
            "training_time_seconds" not in self.df.columns
            or "avg_reward" not in self.df.columns
        ):
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Training time vs reward
        axes[0].scatter(
            self.df["training_time_seconds"] / 3600, self.df["avg_reward"], alpha=0.7
        )
        axes[0].set_xlabel("Training Time (hours)")
        axes[0].set_ylabel("Average Reward")
        axes[0].set_title("Training Efficiency: Time vs Performance")

        # Training time distribution
        axes[1].hist(
            self.df["training_time_seconds"] / 3600,
            bins=15,
            alpha=0.7,
            edgecolor="black",
        )
        axes[1].set_xlabel("Training Time (hours)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Training Times")
        axes[1].axvline(
            self.df["training_time_seconds"].mean() / 3600,
            color="red",
            linestyle="--",
            label=f"Mean: {self.df['training_time_seconds'].mean() / 3600:.2f}h",
        )
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "training_efficiency.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_correlation_heatmap(self, output_dir: str):
        """Plot correlation heatmap of hyperparameters and performance."""
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return

        correlation_matrix = self.df[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
        )
        plt.title("Correlation Matrix of Hyperparameters and Performance")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "correlation_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_policy_comparison(self, output_dir: str):
        """Compare performance between different policies."""
        if "policy_type" not in self.df.columns or "avg_reward" not in self.df.columns:
            return

        if self.df["policy_type"].nunique() < 2:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot comparison
        sns.boxplot(data=self.df, x="policy_type", y="avg_reward", ax=axes[0])
        axes[0].set_title("Performance Comparison by Policy Type")
        axes[0].set_ylabel("Average Reward")

        # Violin plot for distribution comparison
        sns.violinplot(data=self.df, x="policy_type", y="avg_reward", ax=axes[1])
        axes[1].set_title("Performance Distribution by Policy Type")
        axes[1].set_ylabel("Average Reward")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "policy_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def export_summary_table(self, output_file: str = "experiments/summary_table.csv"):
        """Export a clean summary table for reporting."""
        if self.df is None:
            print("❌ No data to export")
            return

        # Select key columns for summary
        summary_cols = [
            "experiment_name",
            "policy_type",
            "learning_rate",
            "gamma",
            "batch_size",
            "epsilon_start",
            "epsilon_end",
            "avg_reward",
            "std_reward",
            "training_time_seconds",
        ]

        # Keep only available columns
        available_cols = [col for col in summary_cols if col in self.df.columns]

        summary_df = self.df[available_cols].copy()

        # Round numeric values for readability
        if "avg_reward" in summary_df.columns:
            summary_df["avg_reward"] = summary_df["avg_reward"].round(2)
        if "std_reward" in summary_df.columns:
            summary_df["std_reward"] = summary_df["std_reward"].round(2)
        if "training_time_seconds" in summary_df.columns:
            summary_df["training_time_hours"] = (
                summary_df["training_time_seconds"] / 3600
            ).round(2)

        # Sort by performance
        if "avg_reward" in summary_df.columns:
            summary_df = summary_df.sort_values("avg_reward", ascending=False)

        summary_df.to_csv(output_file, index=False)
        print(f"📋 Summary table exported to {output_file}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze DQN experiment results")
    parser.add_argument("--member_name", help="Filter by team member name")
    parser.add_argument(
        "--policy", choices=["CNN", "MLP"], help="Filter by policy type"
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--output_dir", default="experiments/plots", help="Output directory for plots"
    )

    args = parser.parse_args()

    print("🔍 DQN Experiment Results Analyzer")
    print("=" * 50)

    # Initialize analyzer
    analyzer = ExperimentAnalyzer()

    if analyzer.df is None:
        print("\n❌ No data to analyze. Make sure you have:")
        print("   1. Run some experiments using train.py or run_experiments.py")
        print("   2. Check that experiments/hyperparameter_results.csv exists")
        return

    # Generate summary report
    print("\n📊 Generating analysis report...")
    report = analyzer.generate_summary_report(args.member_name, args.policy)
    print(report)

    # Save report to file
    report_file = "experiments/analysis_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\n📄 Full report saved to {report_file}")

    # Generate visualizations
    if not args.no_plots:
        print(f"\n📊 Generating visualizations...")
        analyzer.create_visualizations(args.output_dir)

    # Export summary table
    analyzer.export_summary_table()

    print(f"\n✅ Analysis complete! Check the experiments/ directory for results.")


if __name__ == "__main__":
    main()
