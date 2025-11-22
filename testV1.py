import pandas as pd
import numpy as np


def is_corrupted_row(row):
    """
    Detect if a row is corrupted by checking multiple conditions.

    Returns: True if row is corrupted, False otherwise
    """
    issues = []

    # Check 1: Label must be one of the valid categories
    valid_labels = ['news', 'sport', 'commentisfree', 'culture']
    if pd.notna(row.get('label')):
        if row['label'] not in valid_labels:
            issues.append("Invalid label")
    else:
        issues.append("Missing label")

    # Check 2: Body text should be reasonably long
    body_text = str(row.get('body_text', ''))
    if len(body_text) < 20:
        issues.append("Body text too short")

    # Check 3: Body text shouldn't contain CSV artifacts
    # (like having category names or dates in wrong format)
    if body_text.startswith(('news', 'sport', 'commentisfree', 'culture')):
        issues.append("Body text starts with category name")

    # Check 4: Section should be a simple string, not long text
    section = str(row.get('section', ''))
    if len(section) > 100:  # Section names should be short
        issues.append("Section field too long")

    # Check 5: Date should follow a date pattern
    date = str(row.get('date', ''))
    if pd.notna(row.get('date')) and len(date) > 0:
        # Date should be around 10 chars (YYYY-MM-DD format)
        if len(date) > 20 or len(date) < 8:
            issues.append("Date format suspicious")

    # Check 6: URL should contain "theguardian.com"
    url = str(row.get('url', ''))
    if 'theguardian.com' not in url and 'https://' not in url:
        issues.append("Invalid URL")

    # Check 7: Byline shouldn't be extremely long
    byline = str(row.get('byline', ''))
    if len(byline) > 200:  # Author names shouldn't be this long
        issues.append("Byline too long")

    # Check 8: Headline shouldn't be missing or extremely long
    headline = str(row.get('headline', ''))
    if len(headline) < 5:
        issues.append("Headline too short")
    elif len(headline) > 500:
        issues.append("Headline too long")

    if issues:
        return True, issues
    return False, []


def clean_corrupted_data(input_file='sensed_data.csv', output_file='cleaned_data.csv'):
    """
    Clean the CSV file by removing corrupted rows.
    """
    print("=" * 60)
    print("CLEANING CORRUPTED DATA")
    print("=" * 60)

    # Load data
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')
    initial_count = len(df)
    print(f"Initial rows: {initial_count}")

    # Identify corrupted rows
    print("\nScanning for corrupted rows...")
    corrupted_indices = []
    corruption_reasons = {}

    for idx, row in df.iterrows():
        is_corrupt, issues = is_corrupted_row(row)
        if is_corrupt:
            corrupted_indices.append(idx)
            corruption_reasons[idx] = issues

    print(f"\nFound {len(corrupted_indices)} corrupted rows")

    # Show examples of corrupted rows
    if len(corrupted_indices) > 0:
        print("\nExamples of corruption:")
        for idx in corrupted_indices[:5]:  # Show first 5
            print(f"\nRow {idx}:")
            print(f"  Issues: {', '.join(corruption_reasons[idx])}")
            print(f"  Label: {df.loc[idx, 'label']}")
            print(f"  Headline: {str(df.loc[idx, 'headline'])[:80]}...")
            print(f"  Body preview: {str(df.loc[idx, 'body_text'])[:100]}...")

    # Remove corrupted rows AND their neighbors
    print("\n" + "=" * 60)
    print("REMOVING CORRUPTED ROWS + NEIGHBORS")
    print("=" * 60)

    indices_to_remove = set()

    for idx in corrupted_indices:
        # Add the corrupted row itself
        indices_to_remove.add(idx)

        # Add previous row (if exists)
        if idx > 0:
            indices_to_remove.add(idx - 1)

        # Add next row (if exists)
        if idx < len(df) - 1:
            indices_to_remove.add(idx + 1)

    print(f"\nRemoving {len(indices_to_remove)} rows total")
    print(f"  - {len(corrupted_indices)} corrupted rows")
    print(f"  - ~{len(indices_to_remove) - len(corrupted_indices)} neighboring rows")

    # Create clean dataframe
    df_clean = df.drop(index=list(indices_to_remove))
    df_clean = df_clean.reset_index(drop=True)

    final_count = len(df_clean)
    removed_count = initial_count - final_count

    print(f"\nResults:")
    print(f"  Initial rows: {initial_count}")
    print(f"  Removed rows: {removed_count} ({removed_count / initial_count * 100:.1f}%)")
    print(f"  Clean rows: {final_count} ({final_count / initial_count * 100:.1f}%)")

    # Validate the clean data
    print("\n" + "=" * 60)
    print("VALIDATING CLEAN DATA")
    print("=" * 60)

    # Check for any remaining issues
    remaining_issues = 0
    for idx, row in df_clean.iterrows():
        is_corrupt, _ = is_corrupted_row(row)
        if is_corrupt:
            remaining_issues += 1

    if remaining_issues > 0:
        print(f"⚠️  Warning: {remaining_issues} rows still have issues")
        print("You may need to run this again or adjust detection criteria")
    else:
        print("✓ All rows passed validation!")

    # Show class distribution
    print("\nClass distribution in clean data:")
    print(df_clean['label'].value_counts())

    # Save clean data
    df_clean.to_csv(output_file, index=False, encoding='utf-8', quoting=1)
    print(f"\n✓ Clean data saved to: {output_file}")

    return df_clean


def inspect_specific_rows(input_file='sensed_data.csv', start_row=8970, end_row=8985):
    """
    Inspect specific rows to understand corruption patterns.
    """
    print("=" * 60)
    print(f"INSPECTING ROWS {start_row} TO {end_row}")
    print("=" * 60)

    df = pd.read_csv(input_file, encoding='utf-8')

    for idx in range(start_row, min(end_row, len(df))):
        print(f"\n--- Row {idx} ---")
        row = df.iloc[idx]
        is_corrupt, issues = is_corrupted_row(row)

        print(f"Corrupted: {is_corrupt}")
        if is_corrupt:
            print(f"Issues: {', '.join(issues)}")

        print(f"Label: '{row.get('label', 'N/A')}'")
        print(f"Section: '{str(row.get('section', 'N/A'))[:50]}...'")
        print(f"Date: '{row.get('date', 'N/A')}'")
        print(f"Headline: '{str(row.get('headline', 'N/A'))[:80]}...'")
        print(f"Body length: {len(str(row.get('body_text', '')))} chars")
        print(f"Body preview: '{str(row.get('body_text', 'N/A'))[:100]}...'")


if __name__ == "__main__":
    # Option 1: Inspect specific problem area (from your screenshot)
    print("Step 1: Inspecting problem area...")
    inspect_specific_rows(start_row=8970, end_row=8985)

    print("\n" + "=" * 80 + "\n")

    # Option 2: Clean the entire dataset
    print("Step 2: Cleaning entire dataset...")
    df_clean = clean_corrupted_data(
        input_file='sensed_data.csv',
        output_file='cleaned_data.csv'
    )

    print("\n" + "=" * 80)
    print("CLEANING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review 'cleaned_data.csv'")
    print("2. If satisfied, rename it to 'sensed_data.csv'")
    print("3. Or adjust detection criteria and run again")