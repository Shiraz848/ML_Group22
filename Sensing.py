import pandas as pd
from pathlib import Path
import re


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
        section = header_parts[1].strip() if len(header_parts) > 1 else ""
        date = header_parts[2].strip() if len(header_parts) > 2 else ""

        # Parse byline: "By Author Name"
        byline = byline_line.replace("By ", "").strip()

        # URL is on line 3
        url = url_line.strip()

        # Headline is on line 5 (after empty line 4)
        headline = lines[4].strip() if len(lines) > 4 else ""

        # Body text starts after the separator line (line 6)
        # Join all remaining lines and clean up all whitespace (newlines, tabs, etc.)
        raw_text = ' '.join(lines[6:]) if len(lines) > 6 else ""
        body_text = re.sub(r'\s+', ' ', raw_text).strip()

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


def create_sensed_data_csv(output_file="sensed_data.csv"):
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

    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8', lineterminator='\n')

    print(f"\nSensing complete!")
    print(f"Total articles processed: {len(df)}")
    print(f"Articles per category:")
    print(df['label'].value_counts())
    print(f"\nData saved to: {output_file}")

    # Clean the CSV file from broken lines
    clean_csv_file(output_file)

    return df


def clean_csv_file(file_path):
    """
    Post-process the CSV to remove broken lines.
    Identifies lines where the first column (label) is not valid,
    and removes that line AND the preceding line.
    """
    print("\nRunning CSV cleanup...")
    # Added 'label' back to preserve header, handle BOM later
    valid_labels = {'news', 'sport', 'commentisfree', 'culture', 'label'}
    
    path = Path(file_path)
    if not path.exists():
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        lines_to_skip = set()
        
        for i, line in enumerate(lines):
            # Split by comma to check the first column
            parts = line.split(',')
            if not parts:
                continue
            
            # Clean up the first column value: strip whitespace, quotes, and BOM
            first_col = parts[0].strip().replace('"', '').replace('\ufeff', '')
            
            # If the first column is not a known label
            if first_col not in valid_labels:
                # Mark this line for deletion (it's a broken fragment)
                lines_to_skip.add(i)
                # Mark the previous line for deletion (it's the incomplete parent of the fragment)
                if i > 0:
                    lines_to_skip.add(i - 1)
        
        if not lines_to_skip:
            print("No broken lines found.")
            return

        print(f"Found {len(lines_to_skip)} lines to remove (broken lines + predecessors).")
        
        # Write back only the good lines
        with open(path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lines):
                if i not in lines_to_skip:
                    f.write(line)
                    
        print("Cleanup complete.")
            
    except Exception as e:
        print(f"Error during CSV cleanup: {e}")


if __name__ == "__main__":
    # Create the sensed data CSV
    df = create_sensed_data_csv()

    # Display first few rows as preview
    if df is not None:
        print("\nPreview of sensed_data.csv:")
        print(df.head())