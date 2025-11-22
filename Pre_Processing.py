import pandas as pd
import numpy as np
import re
from collections import Counter


def load_sensed_data(file_path="sensed_data.csv"):
    """Load the sensed data CSV."""
    print("Loading sensed data...")
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"Loaded {len(df)} articles")
    return df


def check_missing_values(df):
    """Check and report missing values."""
    print("\n" + "=" * 60)
    print("1. CHECKING MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    })

    print(missing_df[missing_df['Missing Count'] > 0])

    if missing.sum() == 0:
        print("✓ No missing values found!")
    else:
        print(f"\n⚠️  Total missing values: {missing.sum()}")

    return df


def handle_missing_values(df):
    """Handle missing values - fill or drop."""
    print("\nHandling missing values...")

    # For text fields, fill with empty string
    text_columns = ['section', 'byline', 'headline', 'body_text']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Drop rows where body_text is empty (main content missing)
    initial_len = len(df)
    df = df[df['body_text'].str.strip() != '']
    dropped = initial_len - len(df)

    if dropped > 0:
        print(f"Dropped {dropped} articles with empty body text")

    return df


def check_duplicates(df):
    """Check for duplicate articles."""
    print("\n" + "=" * 60)
    print("2. CHECKING DUPLICATES")
    print("=" * 60)

    # Check duplicates by URL (most reliable)
    url_duplicates = df.duplicated(subset=['url'], keep=False).sum()
    print(f"Duplicate URLs: {url_duplicates}")

    # Check duplicates by headline
    headline_duplicates = df.duplicated(subset=['headline'], keep=False).sum()
    print(f"Duplicate headlines: {headline_duplicates}")

    # Check exact duplicates (all columns)
    exact_duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows: {exact_duplicates}")

    return df


def remove_duplicates(df):
    """Remove duplicate articles."""
    print("\nRemoving duplicates...")

    initial_len = len(df)

    # Remove duplicates based on URL (keep first occurrence)
    df = df.drop_duplicates(subset=['url'], keep='first')

    dropped = initial_len - len(df)
    print(f"Removed {dropped} duplicate articles")
    print(f"Remaining articles: {len(df)}")

    return df


def clean_text(text):
    """Clean text by removing/normalizing problematic characters."""
    if not isinstance(text, str):
        return ""

    # Fix common encoding issues
    text = text.replace('â€™', "'")  # apostrophe
    text = text.replace('â€œ', '"')  # left quote
    text = text.replace('â€', '"')  # right quote
    text = text.replace('â€"', '-')  # dash
    text = text.replace('â€¢', '•')  # bullet

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def clean_all_text_fields(df):
    """Apply text cleaning to all text columns."""
    print("\n" + "=" * 60)
    print("3. CLEANING TEXT")
    print("=" * 60)

    text_columns = ['section', 'byline', 'headline', 'body_text']

    for col in text_columns:
        if col in df.columns:
            print(f"Cleaning {col}...")
            df[col] = df[col].apply(clean_text)

    print("✓ Text cleaning complete!")
    return df


def check_class_balance(df):
    """Check distribution of articles across categories."""
    print("\n" + "=" * 60)
    print("4. CHECKING CLASS BALANCE")
    print("=" * 60)

    class_counts = df['label'].value_counts()
    class_pct = (class_counts / len(df)) * 100

    balance_df = pd.DataFrame({
        'Category': class_counts.index,
        'Count': class_counts.values,
        'Percentage': class_pct.values
    })

    print(balance_df)

    # Check if imbalanced
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 2:
        print("⚠️  Data is imbalanced! Consider:")
        print("   - Collecting more data for minority classes")
        print("   - Using stratified sampling in validation")
        print("   - Using class weights in model training")
    else:
        print("✓ Data is relatively balanced")

    return df


def check_text_length_statistics(df):
    """Check statistics about text lengths."""
    print("\n" + "=" * 60)
    print("5. TEXT LENGTH STATISTICS")
    print("=" * 60)

    # Calculate word counts
    df['word_count'] = df['body_text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['body_text'].apply(len)

    print("\nBody Text Statistics:")
    print(f"Average words per article: {df['word_count'].mean():.0f}")
    print(f"Median words per article: {df['word_count'].median():.0f}")
    print(f"Min words: {df['word_count'].min()}")
    print(f"Max words: {df['word_count'].max()}")

    # Check for very short articles (potential issues)
    short_articles = df[df['word_count'] < 50]
    if len(short_articles) > 0:
        print(f"\n⚠️  Found {len(short_articles)} articles with <50 words (may be incomplete)")

    return df


def remove_short_articles(df, min_words=50):
    """Remove articles that are too short."""
    print(f"\nRemoving articles with <{min_words} words...")

    initial_len = len(df)
    df = df[df['word_count'] >= min_words]
    dropped = initial_len - len(df)

    print(f"Removed {dropped} short articles")
    print(f"Remaining articles: {len(df)}")

    return df


def validate_data_quality(df):
    """Final validation checks."""
    print("\n" + "=" * 60)
    print("6. DATA QUALITY VALIDATION")
    print("=" * 60)

    issues = []

    # Check for empty body texts
    empty_body = df[df['body_text'].str.strip() == '']
    if len(empty_body) > 0:
        issues.append(f"{len(empty_body)} articles with empty body text")

    # Check for missing labels
    missing_labels = df[df['label'].isnull() | (df['label'] == '')]
    if len(missing_labels) > 0:
        issues.append(f"{len(missing_labels)} articles with missing labels")

    # Check for very long articles (potential errors)
    very_long = df[df['word_count'] > 10000]
    if len(very_long) > 0:
        issues.append(f"{len(very_long)} articles with >10,000 words (check for errors)")

    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✓ All validation checks passed!")

    return df


def save_preprocessed_data(df, output_file="preprocessed_data.csv"):
    """Save the preprocessed data."""
    print("\n" + "=" * 60)
    print("SAVING PREPROCESSED DATA")
    print("=" * 60)

    # Drop temporary columns
    columns_to_drop = ['word_count', 'char_count']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8', quoting=1)

    print(f"✓ Saved preprocessed data to: {output_file}")
    print(f"Final dataset size: {len(df)} articles")
    print(f"Columns: {list(df.columns)}")

    return df


def preprocess_pipeline(input_file="sensed_data.csv", output_file="preprocessed_data.csv"):
    """Complete preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load data
    df = load_sensed_data(input_file)

    # Step 1: Handle missing values
    df = check_missing_values(df)
    df = handle_missing_values(df)

    # Step 2: Remove duplicates
    df = check_duplicates(df)
    df = remove_duplicates(df)

    # Step 3: Clean text
    df = clean_all_text_fields(df)

    # Step 4: Check class balance
    df = check_class_balance(df)

    # Step 5: Check text statistics
    df = check_text_length_statistics(df)
    df = remove_short_articles(df, min_words=50)

    # Step 6: Validate
    df = validate_data_quality(df)

    # Save
    df = save_preprocessed_data(df, output_file)

    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    # Run the preprocessing pipeline
    df = preprocess_pipeline()