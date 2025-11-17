from pathlib import Path

base_dir = Path("data")
categories = ["news", "sport", "opinion", "culture"]
total = 0

print("🗂️  Number of articles per category:\n")
for category in categories:
    cat_dir = base_dir / category
    num_files = len(list(cat_dir.glob("*.txt")))
    total += num_files
    print(f"{category.capitalize():<10}: {num_files} articles")

print(f"\n📊 Total articles: {total}")