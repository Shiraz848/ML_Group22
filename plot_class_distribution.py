import pandas as pd
import matplotlib.pyplot as plt

def plot_category_distribution(csv_path="preprocessed_data.csv", title="Category Distribution", save=False):
    # Load data
    df = pd.read_csv(csv_path, encoding='utf-8')

    # Count articles per label
    label_counts = df['label'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(label_counts.index, label_counts.values, color='skyblue', edgecolor='black')

    # Add counts on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 50, str(int(height)),
                 ha='center', va='bottom', fontsize=10)

    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Number of Articles")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save:
        plt.savefig("category_distribution.png")
        print("📁 Saved to: category_distribution.png")

    plt.show()

if __name__ == "__main__":
    plot_category_distribution("preprocessed_data.csv", title="Final Category Distribution", save=True)
