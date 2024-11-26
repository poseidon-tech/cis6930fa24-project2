import sys
import os
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unredactor import clean_and_tokenize,extract_features,read_data,process_dataset_features,train,evaluation,predict_test_tsv

def test_read_data_file_not_found():
    non_existent_file = "non_existent_file.tsv" 
    if os.path.exists(non_existent_file):
        os.remove(non_existent_file)
    with pytest.raises(FileNotFoundError):
        read_data(non_existent_file)

def test_clean_and_tokenize():
    text = "The first █████████ movie i've ever seen was breaking the waves. Sure a nice movie but it definitely stands in the shadow of europa."
    tokens = clean_and_tokenize(text)
    assert tokens == ['The', 'first', '@$@', 'movie', 'i', "'ve", 'ever', 'seen', 'was', 'breaking', 'the', 'waves', '.', 'Sure', 'a', 'nice', 'movie', 'but', 'it', 'definitely', 'stands', 'in', 'the', 'shadow', 'of', 'europa', '.']


def test_extract_features():
    train_data, _ = read_data('test_data.tsv')
    row = train_data.iloc[0]
    expected_result = extract_features(row)
    actual_result = {
        'prev_tfidf_mean': np.float64(0.05892556509887895),
        'next_tfidf_mean': np.float64(0.05892556509887895),
        'prev_three_words': 'excellent , especially',
        'prev_three_pos': 'JJ , RB',
        'next_three_words': ', whose final',
        'next_three_pos': ', WP$ JJ',
        'redaction_length': 11,
        'sentiment_compound': 0.8648,
        'person_entities_count': 0
    }
    for key in actual_result:
        if isinstance(actual_result[key], float):  # Handle floating-point comparison
            assert np.isclose(expected_result[key], actual_result[key], atol=1e-6), f"Mismatch in {key}"
        else:
            assert expected_result[key] == actual_result[key], f"Mismatch in {key}: {expected_result[key]} != {actual_result[key]}"

def test_process_dataset_features():
    train_data, val_data = read_data('test_data.tsv')
    X_train,y_train,X_val,y_val,vec = process_dataset_features(train_data,val_data)
    assert X_train.shape == (1, 9), f"Expected shape (1, 9), got {X_train.shape}"
    assert X_train[0, 0] == 0.05892556509887895, "Incorrect value in feature 0"
    assert np.isclose(X_train[0, 7], 11.0, atol=1e-6), "Incorrect redaction length"
    assert np.isclose(X_train[0, 8], 0.8648, atol=1e-6), "Incorrect sentiment score"
    assert len(y_train) == 1, f"Expected length 1, got {len(y_train)}"
    assert y_train[0] == "Ivan Trojan", f"Expected 'Ivan Trojan', got {y_train[0]}"
    assert vec is not None, "DictVectorizer not initialized"

def test_train():
    X_train, y_train = make_classification(n_samples=100, n_features=5, random_state=42)
    X_val, y_val = make_classification(n_samples=20, n_features=5, random_state=24)
    pipeline, y_pred = train(X_train, y_train, X_val, y_val)
    assert isinstance(pipeline, Pipeline), "The returned pipeline is not a scikit-learn Pipeline"
    assert len(y_pred) == len(y_val), f"Mismatch between predictions ({len(y_pred)}) and validation labels ({len(y_val)})"
    
