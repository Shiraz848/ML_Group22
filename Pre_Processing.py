import pandas as pd

def preprocess_articles(input_file="sensed_data.csv", output_file="preprocessed_data.csv"):
    """
    Preprocessing pipeline for cleaning and preparing raw article data.
    Steps include:
    1. Label normalization
    2. Removal of rows with missing critical fields
    3. Removal of duplicate articles
    4. Filtering out correction/clarification articles
    5. Displaying class distribution
    """

    print("Loading data...")
    df = pd.read_csv(input_file, encoding="utf-8")
    print("Initial dataset size:", len(df))

    # -------------------------------------------------------------
    # 1. Normalize label values (lowercase + strip whitespace)
    # -------------------------------------------------------------
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Keep only valid label categories
    valid_labels = ["news", "sport", "commentisfree", "culture"]
    df = df[df["label"].isin(valid_labels)]

    # -------------------------------------------------------------
    # 2. Remove rows with missing critical fields
    # -------------------------------------------------------------
    required_fields = ["headline", "body_text", "url", "label"]
    before_missing = len(df)

    df.dropna(subset=required_fields, inplace=True)
    df = df[df["body_text"].str.strip() != ""]

    print("Rows removed due to missing critical fields:", before_missing - len(df))

    # -------------------------------------------------------------
    # 3. Remove duplicated articles (based on URL)
    # -------------------------------------------------------------
    before_dups = len(df)
    df.drop_duplicates(subset=["url"], inplace=True)
    print("Duplicate articles removed:", before_dups - len(df))

    # -------------------------------------------------------------
    # 4. Remove correction/clarification articles (noise)
    # -------------------------------------------------------------
    before_corr = len(df)

    correction_mask = ~df["headline"].str.lower().str.contains("correction|clarification") & \
                      ~df["body_text"].str.lower().str.contains("correction|clarification")

    df = df[correction_mask]

    print("Correction/clarification articles removed:", before_corr - len(df))

    # -------------------------------------------------------------
    # 5. Display class distribution (to identify imbalance)
    # -------------------------------------------------------------
    print("\nClass Distribution (%):")
    class_distribution = (df["label"].value_counts(normalize=True) * 100).round(2)
    print(class_distribution)

    # -------------------------------------------------------------
    # 6. Save preprocessed dataset
    # -------------------------------------------------------------
    df.to_csv(output_file, index=False, encoding="utf-8")
    print("\nPreprocessing complete.")
    print("Final dataset size:", len(df))
    print("Cleaned data saved to:", output_file)

    return df


if __name__ == "__main__":
    preprocess_articles()
