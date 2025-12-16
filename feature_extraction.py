import pandas as pd
import re
import string
import numpy as np

# --- 1. הגדרת רשימות המילים כקבוצות נושאים ---
WORD_LISTS = {
    'score_politics': [
        'trump', 'biden', 'government', 'police', 'law', 'court', 'minister',
        'president', 'parliament', 'election', 'vote', 'campaign', 'senate',
        'congress', 'brexit', 'eu', 'un', 'nato', 'border', 'policy', 'party'
    ],
    'score_conflict': [
        'crisis', 'war', 'military', 'attack', 'security', 'official', 'protest',
        'strike', 'bomb', 'peace', 'weapon', 'refugee', 'army', 'soldier'
    ],
    'score_economy': [
        'economy', 'tax', 'bank', 'budget', 'market', 'money', 'price', 'cost',
        'trade', 'inflation', 'business', 'finance', 'debt'
    ],
    'score_sport': [
        'goal', 'match', 'team', 'win', 'score', 'league', 'player', 'coach',
        'tournament', 'championship', 'cup', 'final', 'manager', 'referee',
        'stadium', 'ball', 'race', 'medal', 'olympic', 'wimbledon', 'tennis',
        'football', 'cricket', 'nba', 'season', 'injury', 'title', 'defeat', 'victory'
    ],
    'score_culture': [
        'film', 'movie', 'cinema', 'hollywood', 'director', 'actor', 'actress',
        'star', 'theatre', 'drama', 'comedy', 'show', 'series', 'episode',
        'netflix', 'award', 'oscar', 'plot', 'character',
        'book', 'novel', 'author', 'writer', 'publish', 'fiction', 'art',
        'artist', 'exhibition', 'gallery', 'museum', 'painting', 'sculpture',
        'music', 'song', 'album', 'band', 'singer', 'concert', 'performance',
        'rock', 'pop', 'sound'
    ],
    'score_opinion': [
        'opinion', 'comment', 'editorial', 'column', 'perspective', 'view',
        'believe', 'think', 'feel', 'seem', 'appear', 'suggest', 'argue',
        'claim', 'perhaps', 'maybe', 'probably', 'surely', 'clearly', 'doubt',
        'question', 'hope', 'fear', 'must', 'should',
        'i', 'my', 'me', 'we', 'our', 'us'
    ]
}


def extract_features(row):
    # שליפת נתונים בטוחה
    title = str(row['headline']) if pd.notna(row.get('headline')) else ""
    body = str(row['body_text']) if pd.notna(row.get('body_text')) else ""
    byline = str(row['byline']) if 'byline' in row and pd.notna(row['byline']) else ""
    date_str = str(row['date'])[:10] if 'date' in row and pd.notna(row['date']) else ""

    features = {}

    # הכנה
    body_words = body.split()
    num_words = len(body_words)
    if num_words == 0: num_words = 1

    # --- פיצ'רים מבניים ---
    features['title_length'] = len(title)
    features['num_words'] = num_words
    features['num_sentences'] = body.count('.') + body.count('!') + body.count('?')
    features['num_paragraphs'] = body.count('\n') + 1

    # --- פיצ'רים יחסיים (Ratios) ---
    features['avg_sentence_length'] = num_words / max(1, features['num_sentences'])
    features['avg_word_length'] = sum(len(w) for w in body_words) / num_words
    features['ratio_long_words'] = sum(1 for w in body_words if len(w) > 6) / num_words
    features['ratio_quotes'] = (body.count('"') + body.count('“')) / num_words
    features['ratio_questions'] = body.count('?') / num_words
##########################################################################
    features['lexical_diversity'] = len(set(body_words)) / num_words
##########################################################################

    # --- מטא-דאטה (כולל התיקון למספר מחברים) ---
    if len(byline.strip()) > 0:
        features['has_author'] = 1
        # ספירת המילה ' and ' היא אינדיקציה טובה למספר כותבים באנגלית (John Doe and Jane Smith)
        features['num_authors'] = byline.lower().count(' and ') + 1
    else:
        features['has_author'] = 0
        features['num_authors'] = 0

    # תאריך (סופ"ש)
    features['is_weekend'] = 0
    try:
        if date_str:
            dt = pd.to_datetime(date_str, errors='coerce')
            if pd.notnull(dt):
                features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
    except:
        pass

    # --- ציוני נושאים (Scores) ---
    body_lower = body.lower()
    for category, words in WORD_LISTS.items():
        count = 0
        for word in words:
            count += len(re.findall(r'\b' + re.escape(word) + r'\b', body_lower))
        features[category] = count / num_words

    return pd.Series(features)


def run_feature_extraction(input_file='segmentation_data.csv', output_file='features_data.csv'):
    print(f"📥 Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found.")
        return

    print("🔎 Extracting features...")

    feature_df = df.apply(extract_features, axis=1)

    if 'label' in df.columns:
        feature_df['label'] = df['label']
    elif 'category' in df.columns:
        feature_df['label'] = df['category']

    feature_df = feature_df.fillna(0)

    print(f"💾 Saving features to: {output_file}")
    feature_df.to_csv(output_file, index=False)

    print("\n✅ Feature Extraction Complete.")
    print(f"Total articles: {len(feature_df)}")
    print("Columns created:", list(feature_df.columns))


if __name__ == "__main__":
    run_feature_extraction()





# import pandas as pd
# import re
# import string
# import numpy as np
# from datetime import datetime
#
# # --- 1. הגדרת רשימות המילים (כולל commentisfree ו-Trump) ---
# WORD_LISTS = {
#     'news_politics': [
#         'trump', 'biden', 'government', 'police', 'law', 'court', 'minister',
#         'president', 'parliament', 'election', 'vote', 'campaign', 'senate',
#         'congress', 'brexit', 'eu', 'un', 'nato', 'border', 'policy', 'party'
#     ],
#     'news_conflict': [
#         'crisis', 'war', 'military', 'attack', 'security', 'official', 'protest',
#         'strike', 'bomb', 'peace', 'weapon', 'refugee', 'army', 'soldier'
#     ],
#     'news_economy': [
#         'economy', 'tax', 'bank', 'budget', 'market', 'money', 'price', 'cost',
#         'trade', 'inflation', 'business', 'finance', 'debt'
#     ],
#     'sport': [
#         'goal', 'match', 'team', 'win', 'score', 'league', 'player', 'coach',
#         'tournament', 'championship', 'cup', 'final', 'manager', 'referee',
#         'stadium', 'ball', 'race', 'medal', 'olympic', 'wimbledon', 'tennis',
#         'football', 'cricket', 'nba', 'season', 'injury', 'title', 'defeat', 'victory'
#     ],
#     'culture_movies': [
#         'film', 'movie', 'cinema', 'hollywood', 'director', 'actor', 'actress',
#         'star', 'theatre', 'drama', 'comedy', 'show', 'series', 'episode',
#         'netflix', 'award', 'oscar', 'plot', 'character'
#     ],
#     'culture_books_art': [
#         'book', 'novel', 'author', 'writer', 'publish', 'fiction', 'art',
#         'artist', 'exhibition', 'gallery', 'museum', 'painting', 'sculpture'
#     ],
#     'culture_music': [
#         'music', 'song', 'album', 'band', 'singer', 'concert', 'performance',
#         'rock', 'pop', 'sound', 'listen'
#     ],
#     'commentisfree_general': [
#         'opinion', 'comment', 'editorial', 'column', 'perspective', 'view',
#         'believe', 'think', 'feel', 'seem', 'appear', 'suggest', 'argue',
#         'claim', 'perhaps', 'maybe', 'probably', 'surely', 'clearly', 'doubt',
#         'question', 'hope', 'fear', 'must', 'should'
#     ],
#     'commentisfree_first_person': [
#         'i', 'my', 'me', 'we', 'our', 'us'
#     ]
# }
#
# # --- 2. הכנת תבניות חיפוש (Pre-compiling Regex) ---
# REGEX_PATTERNS = {}
# for category, words in WORD_LISTS.items():
#     for word in words:
#         pattern = re.compile(r'\b' + re.escape(word) + r'\b')
#         REGEX_PATTERNS[f'word_{word}'] = pattern
#
#
# def extract_features(row):
#     # המרת נתונים למחרוזות
#     title = str(row['headline']) if pd.notna(row['headline']) else ""
#     body = str(row['body_text']) if pd.notna(row['body_text']) else ""
#     byline = str(row['byline']) if pd.notna(row['byline']) else ""
#     date_str = str(row['date'])[:10]
#
#     features = {}
#
#     # ---------------------------------------------------------
#     # חלק א': פיצ'רים של אותיות גדולות וסגנון (Case Sensitivity)
#     # ---------------------------------------------------------
#     body_words = body.split()
#     num_words = len(body_words)
#
#     features['num_all_caps_words'] = sum(1 for w in body_words if w.isupper() and len(w) > 1)
#     features['num_title_case_words'] = sum(1 for w in body_words if w[0].isupper())
#     # features['ratio_all_caps'] = features['num_all_caps_words'] / max(1, num_words)
#     # features['title_all_caps_ratio'] = sum(1 for w in title.split() if w.isupper()) / max(1, len(title.split()))
#
#     # ---------------------------------------------------------
#     # חלק ב': פיצ'רים מבניים רגילים
#     # ---------------------------------------------------------
#     features['title_length'] = len(title)
#     features['num_words_in_body'] = num_words
#     # features['avg_word_length'] = sum(len(w) for w in body_words) / max(1, num_words)
#     features['num_sentences'] = body.count('.') + body.count('!') + body.count('?')
#     features['num_paragraphs'] = body.count('\n') + 1
#     features['num_long_words'] = sum(1 for w in body_words if len(w) > 7)
#     features['punctuation_density'] = sum(1 for c in body if c in string.punctuation) / max(1, len(body))
#     features['has_quotes'] = int('"' in body or "'" in body or '“' in body)
#     features['num_digits'] = sum(c.isdigit() for c in body)
#     features['has_author'] = int(len(byline.strip()) > 0)
#     features['num_authors'] = byline.count(' and ') + 1 if len(byline.strip()) > 0 else 0
#
#     # טיפול בתאריך - הוספתי את ה-except שהיה חסר
#     try:
#         dt = pd.to_datetime(date_str, errors='coerce')
#         if pd.notnull(dt):
#             features['month'] = dt.month
#             features['day_of_week'] = dt.weekday()
#             features['is_weekend'] = int(dt.weekday() >= 5)
#         else:
#             features['month'] = -1
#             features['day_of_week'] = -1
#             features['is_weekend'] = -1
#     except:
#         features['month'] = -1
#         features['day_of_week'] = -1
#         features['is_weekend'] = -1
#
#     # ---------------------------------------------------------
#     # חלק ג': פיצ'רים מבוססי מילים (Keywords)
#     # ---------------------------------------------------------
#     body_lower = body.lower()
#
#     for feature_name, pattern in REGEX_PATTERNS.items():
#         features[feature_name] = len(pattern.findall(body_lower))
#
#     return pd.Series(features)
#
#
# def run_feature_extraction(input_file='segmentation_data.csv', output_file='features_data.csv'):
#     print(f"📥 Loading data from {input_file}...")
#     try:
#         df = pd.read_csv(input_file)
#     except FileNotFoundError:
#         print(f"❌ Error: File '{input_file}' not found.")
#         return
#
#     print("🔎 Extracting features...")
#
#     feature_df = df.apply(extract_features, axis=1)
#
#     if 'label' in df.columns:
#         feature_df['label'] = df['label']
#     elif 'category' in df.columns:
#         feature_df['label'] = df['category']
#
#     feature_df = feature_df.fillna(0)
#
#     print(f"💾 Saving features to: {output_file}")
#     feature_df.to_csv(output_file, index=False)
#     print("✅ Done.")
#
#
# if __name__ == "__main__":
#     run_feature_extraction()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # import pandas as pd
# # import re
# # import string
# # from datetime import datetime
# #
# # SPORT_TERMS = ['goal', 'match', 'team', 'win', 'score', 'league', 'player', 'coach', 'tournament']
# #
# #
# # def extract_features(row):
# #     title = str(row['headline'])
# #     body = str(row['body_text'])
# #     byline = str(row['byline'])
# #     date = str(row['date'])[:10]  # yyyy-mm-dd
# #
# #     # Feature calculations
# #     features = {}
# #
# #     features['title_length'] = len(title)
# #     features['num_words_in_body'] = len(body.split())
# #     features['avg_word_length'] = sum(len(w) for w in body.split()) / max(1, len(body.split()))
# #     features['num_sentences'] = body.count('.') + body.count('!') + body.count('?')
# #     features['contains_question_mark'] = int('?' in title)
# #     features['num_uppercase_words'] = sum(1 for w in title.split() if w.isupper())
# #     features['has_author'] = int(len(byline.strip()) > 0)
# #
# #     try:
# #         dt = pd.to_datetime(date)
# #         features['month'] = dt.month
# #         features['day_of_week'] = dt.weekday()  # Monday=0
# #     except:
# #         features['month'] = -1
# #         features['day_of_week'] = -1
# #
# #     features['count_sport_terms'] = sum(body.lower().count(term) for term in SPORT_TERMS)
# #     features['num_paragraphs'] = body.count('\n') + 1
# #     features['num_authors'] = byline.count(' and ') + 1 if byline else 0
# #     features['num_long_words'] = sum(1 for w in body.split() if len(w) > 7)
# #     features['punctuation_density'] = sum(1 for c in body if c in string.punctuation) / max(1, len(body))
# #     features['has_quotes'] = int('"' in body or "'" in body)
# #     features['num_digits'] = sum(c.isdigit() for c in body)
# #
# #     return pd.Series(features)
# #
# #
# # def run_feature_extraction(input_file='preprocessed_data.csv', output_file='features_data.csv'):
# #     print("📥 Loading preprocessed data...")
# #     df = pd.read_csv(input_file)
# #
# #     print("🔎 Extracting features...")
# #     feature_df = df.apply(extract_features, axis=1)
# #
# #     # Add the label column
# #     feature_df['label'] = df['label']
# #
# #     print("💾 Saving features to:", output_file)
# #     feature_df.to_csv(output_file, index=False)
# #     print("✅ Done. Total articles processed:", len(feature_df))
# #
# #     return feature_df
# #
# #
# # if __name__ == "__main__":
# #     run_feature_extraction()
