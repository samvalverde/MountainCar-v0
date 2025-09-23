import os
import pandas as pd
import matplotlib.pyplot as plt

# poetry run python src/experiments/compare_results.py

def compare_results(out_dir="src/experiments/outputs"):
    csv_files = [f for f in os.listdir(out_dir) if f.endswith(".csv")]

    if not csv_files:
        print("⚠️ No CSV files found. Run results.py first.")
        return

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:
        path = os.path.join(out_dir, csv_file)
        df = pd.read_csv(path)

        label = csv_file.replace("fitness_", "").replace(".csv", "")
        plt.plot(df["Generation"], df["Best"], label=f"{label} (Best)")
        plt.plot(df["Generation"], df["Average"], linestyle="--", label=f"{label} (Avg)")

    plt.xlabel("Generation")
    plt.ylabel("Fitness (reward)")
    plt.title("Comparison of GA Configurations - MountainCar-v0")
    plt.legend()
    path_png = os.path.join(out_dir, "comparison.png")
    plt.savefig(path_png)
    plt.close()

    print(f"Comparison plot saved at {path_png}")


if __name__ == "__main__":
    compare_results()
