import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Please place 'Mall_Customers (3).csv' in the project folder."
        )
    return pd.read_csv(data_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["CustomerID", "Gender"], inplace=True, errors="ignore")
    return df


def train_kmeans(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> pd.DataFrame:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["Cluster"] = model.fit_predict(df)
    return df


def plot_elbow(df: pd.DataFrame, output_path: Path | None = None) -> None:
    wcss = []
    for i in range(1, 11):
        model = KMeans(n_clusters=i, random_state=42, n_init=10)
        model.fit(df)
        wcss.append(model.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved elbow plot to: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_clusters(df: pd.DataFrame, output_path: Path | None = None) -> None:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="Age",
        y="Annual Income (k$)",
        hue="Cluster",
        palette="tab10",
        data=df,
        legend="full",
        s=60,
    )
    plt.title("K-Means Clustering: Age vs Annual Income")
    plt.xlabel("Age")
    plt.ylabel("Annual Income (k$)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved cluster plot to: {output_path}")
    else:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train K-Means on the Mall Customers dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "Mall_Customers (3).csv",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Mall_Customers (4).csv",
        help="Output CSV file path with cluster labels",
    )
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--no-plots", action="store_true", help="Do not display plots")
    args = parser.parse_args()

    df = load_data(args.input)
    df = preprocess(df)

    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    if not all(col in df.columns for col in features):
        raise ValueError(
            "Expected dataset columns missing. Ensure the CSV includes Age, Annual Income (k$), and Spending Score (1-100)."
        )

    train_df = train_kmeans(df[features], n_clusters=args.clusters)
    output_df = pd.concat([df.reset_index(drop=True), train_df["Cluster"].reset_index(drop=True)], axis=1)
    output_df.to_csv(args.output, index=False)
    print(f"Saved clustered dataset to: {args.output}")

    if not args.no_plots:
        sns.set_style("whitegrid")
        plot_elbow(df[features])
        plot_clusters(output_df)
        silhouette = silhouette_score(df[features], output_df["Cluster"])
        print(f"Silhouette Score: {silhouette:.4f}")


if __name__ == "__main__":
    main()
