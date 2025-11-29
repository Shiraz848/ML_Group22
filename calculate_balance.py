import pandas as pd

def calculate_articles_to_balance(csv_path="preprocessed_data.csv", target="max"):
    """
    מחשבת כמה כתבות חסרות בכל קטגוריה כדי לאזן את הדאטה:
    - לפי הקטגוריה עם הכי הרבה כתבות (target='max')
    - או לפי ממוצע (target='mean')
    """

    df = pd.read_csv(csv_path)
    label_counts = df["label"].value_counts()

    if target == "max":
        target_count = label_counts.max()
    elif target == "mean":
        target_count = int(label_counts.mean())
    else:
        raise ValueError("target must be 'max' or 'mean'")

    print("\n📊 Current Distribution:")
    print(label_counts)

    print(f"\n🎯 Target per class: {target_count}")
    print("➕ Articles needed to balance:")

    needed = target_count - label_counts
    needed = needed[needed > 0]

    for label, count in needed.items():
        print(f"  - {label}: need {count} more articles")

    return needed


if __name__ == "__main__":
    calculate_articles_to_balance("preprocessed_data.csv", target="max")
