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
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer


# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()
# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
# Load spacy module
nlp = spacy.load("en_core_web_md")


# Function to clean and tokenize text
def clean_and_tokenize(text):
    text = re.sub(r'█+', ' REDACTED_PLACEHOLDER ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [token.replace("REDACTED_PLACEHOLDER", "@$@") for token in tokens]
    return tokens

# Function to extract features
def extract_features(row):
    context = row['context']
    redaction_length = len(row['name']) if 'name' in row else len(row['context'])  # Length fallback for test data
    tokens = clean_and_tokenize(context)
    pos_tags = pos_tag(tokens)
    doc = nlp(context)
    features = {}
    try:
        redaction_index = tokens.index('@$@')
    except ValueError:
        print(f"Redaction block not found in context: {context}")
        redaction_index = math.floor(len(tokens) / 2)

    if redaction_index >= 3:
        prev_three_words = ' '.join(tokens[redaction_index - 3:redaction_index])
        prev_three_pos = ' '.join([pos for word, pos in pos_tags[redaction_index - 3:redaction_index]])
    else:
        prev_three_words = 'NONE'
        prev_three_pos = 'NONE'
    if redaction_index + 3 <= len(tokens):
        next_three_words = ' '.join(tokens[redaction_index + 1:redaction_index + 4])
        next_three_pos = ' '.join([pos for word, pos in pos_tags[redaction_index + 1:redaction_index + 4]])
    else:
        next_three_words = 'NONE'
        next_three_pos = 'NONE'  
    prev_words_tfidf = tfidf_vectorizer.transform([prev_three_words]).toarray().flatten()
    next_words_tfidf = tfidf_vectorizer.transform([next_three_words]).toarray().flatten()
    features['prev_tfidf_mean'] = prev_words_tfidf.mean() if prev_words_tfidf.size > 0 else 0
    features['next_tfidf_mean'] = next_words_tfidf.mean() if next_words_tfidf.size > 0 else 0
    features['prev_three_words'] = prev_three_words
    features['prev_three_pos'] = prev_three_pos
    features['next_three_words'] = next_three_words
    features['next_three_pos'] = next_three_pos
    features['redaction_length'] = redaction_length 
    sentiment = sentiment_analyzer.polarity_scores(context)
    features['sentiment_compound'] = sentiment['compound']
    entities = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    features['person_entities_count'] = len(entities)
    return features

# Read data, preprocess and load it in Data Frame
def read_data(filepath):
    df = pd.read_csv(filepath, sep='\t', names=['split', 'name', 'context'], on_bad_lines='skip', index_col=None, quoting=3, header=0)
    df = df[df['context'].notna() & (df['context'].str.strip() != '')]
    train_data = df[df['split'] == 'training']
    val_data = df[df['split'] == 'validation']
    tfidf_vectorizer.fit(train_data['context'])
    return train_data,val_data

# Extract features 
def process_dataset_features(train_data,val_data):
    # Training data
    X_train = []
    y_train = []
    for _, row in train_data.iterrows():
        features = extract_features(row)
        if features is not None:  # Ensure valid features
            X_train.append(features)
            y_train.append(row['name'])

    # Testing data
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

    return X_train,y_train,X_val,y_val,vec

# Train the model and predict
def train(X_train,y_train,X_val,y_val):
    rf = RandomForestClassifier()
    pipeline = Pipeline([
        ('classifier', rf)
    ])

    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_val)

    return pipeline,y_pred

# Evaluates the trained model
def evaluation(y_val,y_pred):
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

# Reads the test.tsv and predicted output is printed to submission.tsv  
def predict_test_tsv(pipeline,vec):
    #Process test.tsv
    test_data = pd.read_csv('test.tsv', sep='\t', names=['id', 'context'], header=None)
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
    submission = pd.DataFrame({'id': ids, 'name': test_predictions})
    submission.to_csv('submission.tsv', sep='\t', index=False, header=False)
    print("Test predictions saved to submission.tsv")

# Entry Point
def main():
    train_data, val_data = read_data('unredactor.tsv')
    X_train,y_train,X_val,y_val,vec = process_dataset_features(train_data,val_data)
    pipeline,y_pred = train(X_train,y_train,X_val,y_val)
    evaluation(y_val,y_pred)
    predict_test_tsv(pipeline,vec)

if __name__ == "__main__":
    main()