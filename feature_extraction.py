import pandas as pd
import re
import string
from datetime import datetime

SPORT_TERMS = ['goal', 'match', 'team', 'win', 'score', 'league', 'player', 'coach', 'tournament']


def extract_features(row):
    title = str(row['headline'])
    body = str(row['body_text'])
    byline = str(row['byline'])
    date = str(row['date'])[:10]  # yyyy-mm-dd

    # Feature calculations
    features = {}

    features['title_length'] = len(title)
    features['num_words_in_body'] = len(body.split())
    features['avg_word_length'] = sum(len(w) for w in body.split()) / max(1, len(body.split()))
    features['num_sentences'] = body.count('.') + body.count('!') + body.count('?')
    features['contains_question_mark'] = int('?' in title)
    features['num_uppercase_words'] = sum(1 for w in title.split() if w.isupper())
    features['has_author'] = int(len(byline.strip()) > 0)

    try:
        dt = pd.to_datetime(date)
        features['month'] = dt.month
        features['day_of_week'] = dt.weekday()  # Monday=0
    except:
        features['month'] = -1
        features['day_of_week'] = -1

    features['count_sport_terms'] = sum(body.lower().count(term) for term in SPORT_TERMS)
    features['num_paragraphs'] = body.count('\n') + 1
    features['num_authors'] = byline.count(' and ') + 1 if byline else 0
    features['num_long_words'] = sum(1 for w in body.split() if len(w) > 7)
    features['punctuation_density'] = sum(1 for c in body if c in string.punctuation) / max(1, len(body))
    features['has_quotes'] = int('"' in body or "'" in body)
    features['num_digits'] = sum(c.isdigit() for c in body)

    return pd.Series(features)


def run_feature_extraction(input_file='preprocessed_data.csv', output_file='features_data.csv'):
    print("📥 Loading preprocessed data...")
    df = pd.read_csv(input_file)

    print("🔎 Extracting features...")
    feature_df = df.apply(extract_features, axis=1)

    # Add the label column
    feature_df['label'] = df['label']

    print("💾 Saving features to:", output_file)
    feature_df.to_csv(output_file, index=False)
    print("✅ Done. Total articles processed:", len(feature_df))

    return feature_df


if __name__ == "__main__":
    run_feature_extraction()
