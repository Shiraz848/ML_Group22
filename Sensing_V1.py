import pandas as pd
from pathlib import Path
import re


def clean_text(text):
    """
    Clean text by removing/replacing problematic characters.
    """
    # Replace various types of newlines and carriage returns
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    # Remove any other control characters
    text = ''.join(char if char.isprintable() or char.isspace() else ' ' for char in text)
    # Collapse multiple spaces into one
    text = ' '.join(text.split())
    return text


def parse_article_file(file_path):
    """
    Parse a single article txt file and extract its components.

    Returns a dictionary with: section, date, byline, url, headline, body_text, label
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')

        # Extract header information (first 3 lines)
        header_line = lines[0] if len(lines) > 0 else ""
        byline_line = lines[1] if len(lines) > 1 else ""
        url_line = lines[2] if len(lines) > 2 else ""

        # Parse header: "The Guardian | Section | Date"
        header_parts = header_line.split('|')
        section = clean_text(header_parts[1]) if len(header_parts) > 1 else ""
        date = clean_text(header_parts[2]) if len(header_parts) > 2 else ""

        # Parse byline: "By Author Name"
        byline = clean_text(byline_line.replace("By ", ""))

        # URL is on line 3
        url = url_line.strip()

        # Headline is on line 5 (after empty line 4)
        headline = clean_text(lines[4]) if len(lines) > 4 else ""

        # Body text starts after the separator line (line 6)
        # Join all remaining lines and clean thoroughly
        body_text = ' '.join(lines[6:]) if len(lines) > 6 else ""
        body_text = clean_text(body_text)

        # Label is the category folder name
        label = file_path.parent.name

        return {
            'section': section,
            'date': date,
            'byline': byline,
            'url': url,
            'headline': headline,
            'body_text': body_text,
            'label': label
        }

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def collect_all_articles(data_dir="data"):
    """
    Collect all articles from all category folders.

    Returns a list of article dictionaries.
    """
    data_path = Path(data_dir)
    articles = []

    # Categories to process
    categories = ['news', 'sport', 'commentisfree', 'culture']

    for category in categories:
        category_path = data_path / category

        if not category_path.exists():
            print(f"Warning: Category folder '{category}' not found. Skipping.")
            continue

        # Get all txt files in this category
        txt_files = list(category_path.glob('*.txt'))
        print(f"Processing {len(txt_files)} articles from '{category}'...")

        for txt_file in txt_files:
            article_data = parse_article_file(txt_file)
            if article_data:
                articles.append(article_data)

    return articles


def create_sensed_data_csv(output_file="sensed_dataV1.csv"):
    """
    Main function to create the sensed_data.csv file.
    """
    print("Starting sensing process...")

    # Collect all articles
    articles = collect_all_articles()

    if not articles:
        print("No articles found! Please check your data directory.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(articles)

    # Reorder columns for clarity
    column_order = ['label', 'section', 'date', 'byline', 'headline', 'url', 'body_text']
    df = df[column_order]

    # Save to CSV with proper quoting to handle any remaining special characters
    df.to_csv(output_file, index=False, encoding='utf-8', quoting=1)  # quoting=1 is QUOTE_ALL

    print(f"\nSensing complete!")
    print(f"Total articles processed: {len(df)}")
    print(f"Articles per category:")
    print(df['label'].value_counts())
    print(f"\nData saved to: {output_file}")

    return df


if __name__ == "__main__":
    # Create the sensed data CSV
    df = create_sensed_data_csv()

    # Display first few rows as preview
    if df is not None:
        print("\nPreview of sensed_data.csv:")
        print(df.head())