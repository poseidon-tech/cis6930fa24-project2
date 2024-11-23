import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer





# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_md")

# Function to clean and tokenize text
def clean_and_tokenize(text):
    text = re.sub(r'█+', ' █REDACTED█ ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    return tokens

# Function to extract features
def extract_features(row):
    context = row['context']
    redaction_length = len(row['name']) if 'name' in row else len(row['context'])  # Length fallback for test data
    tokens = clean_and_tokenize(context)
    doc = nlp(context)
    features = {}
    try:
        redaction_index = tokens.index('█REDACTED█')
    except ValueError:
        print(f"Redaction block not found in context: {context}")
        redaction_index = math.floor(len(tokens) / 2)

    features['prev_word'] = tokens[redaction_index - 1] if redaction_index > 0 else 'NONE'
    features['next_word'] = tokens[redaction_index + 1] if redaction_index < len(tokens) - 1 else 'NONE'
    features['redaction_length'] = redaction_length
    features['prev_bigram'] = ' '.join(tokens[redaction_index - 2:redaction_index]) if redaction_index > 1 else 'NONE'
    features['next_bigram'] = ' '.join(tokens[redaction_index + 1:redaction_index + 3]) if redaction_index < len(tokens) - 2 else 'NONE'
    sentiment = sentiment_analyzer.polarity_scores(context)
    features['sentiment_compound'] = sentiment['compound']
    entities = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    features['person_entities_count'] = len(entities)
    features['contains_person_entity'] = 1 if len(entities) > 0 else 0
    return features

# Read and preprocess data
df = pd.read_csv('unredactor.tsv', sep='\t', names=['split', 'name', 'context'], on_bad_lines='skip', index_col=None, quoting=3, header=0)
df = df[df['context'].notna() & (df['context'].str.strip() != '')]
train_data = df[df['split'] == 'training']
val_data = df[df['split'] == 'validation']

# Extract features for training data
X_train = []
y_train = []

for _, row in train_data.iterrows():
    features = extract_features(row)
    if features is not None:  # Ensure valid features
        X_train.append(features)
        y_train.append(row['name'])

# Extract features for validation data
X_val = []
y_val = []

for _, row in val_data.iterrows():
    features = extract_features(row)
    if features is not None:  # Ensure valid features
        X_val.append(features)
        y_val.append(row['name'])

# Convert to suitable format using DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train)
X_val = vec.transform(X_val)

# Train the model
rf = RandomForestClassifier()
pipeline = Pipeline([
    ('classifier', rf)
])

pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_val)

# Metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

print(f"Validation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

#Process test.tsv
test_data = pd.read_csv('test.tsv', sep='\t', names=['id', 'context'], header=0)
X_test = []
ids = []

for _, row in test_data.iterrows():
    features = extract_features(row)
    if features is not None:
        X_test.append(features)
        ids.append(row['id'])

X_test = vec.transform(X_test)

# Predict on test data
test_predictions = pipeline.predict(X_test)

# Save predictions to submission.tsv
submission = pd.DataFrame({'id': ids, 'name': test_predictions})
submission.to_csv('submission.tsv', sep='\t', index=False)
print("Test predictions saved to submission.tsv.")