import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_plot(df_path, output_path):
    sns.set_theme(style="whitegrid", context="talk")
    df = pd.read_csv(df_path)

    # Processing the DataFrame
    performance_split = df["Performance (images/sec)"].str.split(" ± ", expand=True)
    df["Mean Performance"] = performance_split[0].astype(float)
    df["Std Dev"] = performance_split[1].astype(float)
    df["Library with Version"] = df["Library"] + ", " + df["Version"]
    df_sorted = df.sort_values("Mean Performance", ascending=True)

    # Create the bar plot
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x="Mean Performance", y="Library with Version", data=df_sorted, palette="viridis")

    # Manually add error bars
    # The positions of bars (center) are usually at half-integers (0.5, 1.5, ...) in seaborn's horizontal barplot
    # But we'll calculate directly from the generated plot to be more precise
    y_positions = [p.get_y() + p.get_height() / 2 for p in barplot.patches]
    error_values = df_sorted["Std Dev"].to_numpy()

    for y_pos, x_val, error_val in zip(y_positions, df_sorted["Mean Performance"], error_values, strict=False):
        plt.errorbar(
            x=x_val,
            y=y_pos,
            xerr=error_val,  # Horizontal error for horizontal bar plot
            fmt="none",  # No connecting lines
            capsize=5,  # Cap size
            color="black",  # Color of the error bars
        )

    # Plot customization
    plt.xlabel("Mean Performance (images/sec)")
    plt.ylabel("")
    plt.title("Library Performance Comparison")
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Create a plot from benchmark results DataFrame")
    parser.add_argument(
        "-f", "--file_path", required=True, help="Path to the CSV file containing the benchmark results"
    )
    parser.add_argument("-o", "--output_path", required=True, help="Path where the plot image will be saved")
    return parser.parse_args()


def main():
    args = parse_args()
    create_plot(args.file_path, args.output_path)


if __name__ == "__main__":
    main()
