import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unredactor import clean_and_tokenize,extract_features,read_data,feature_extraction,train,evaluation,predict_test_tsv

def test_clean_and_tokenize():
    text = "The first █████████ movie i've ever seen was breaking the waves. Sure a nice movie but it definitely stands in the shadow of europa."
    tokens = clean_and_tokenize(text)
    assert tokens == ['The', 'first', '@$@', 'movie', 'i', "'ve", 'ever', 'seen', 'was', 'breaking', 'the', 'waves', '.', 'Sure', 'a', 'nice', 'movie', 'but', 'it', 'definitely', 'stands', 'in', 'the', 'shadow', 'of', 'europa', '.']
