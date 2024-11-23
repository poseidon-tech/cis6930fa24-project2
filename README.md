# CIS6930FA24 -- Project 2: The Unredactor

**Name:** Prajay Yalamanchili

## Project Description

This project is focused on building an Unredactor, a machine learning model designed to predict the most likely names to replace redacted information from text documents. The project involves:

- Training a machine learning pipeline to predict names based on contextual clues.
- Evaluating the model's performance on a validation set using metrics such as accuracy, precision, recall, and F1-score.
- Generating predictions for a provided test dataset and saving them in a specified format.

The redacted content in the dataset is replaced by block characters (`█`). The task is to accurately unredact these blocks, focusing primarily on personal names.

## How to Install

**For Windows:**

1. If Python is not already installed on your system, download and install Python version 3.12 from [here](https://www.python.org/downloads/).
2. Set your path in the environment variables. To learn how to set the path in environment variables, read this [article](https://www.liquidweb.com/help-docs/adding-python-path-to-windows-10-or-11-path-environment-variable/).
3. Download or clone this repository.
4. Navigate to the project directory on your local machine.
5. Run the following commands:

    ```bash
    pip install pipenv
    ```
    ```bash
    pipenv install
    ```

## How to Run

To execute the `main.py` file, use:
```bas
pipenv run python main.py 
```
To run tests, use:
```bash
pipenv run python -m pytest -v
```
or

```bash
pipenv run pytest
```

## Folder Structure

```
|   README.md
|   unredactor.py
|   unredactor.tsv
|   test.tsv
|   submission.tsv
|   requirements.txt
|
+---data
|       Additional data files if necessary
+---docs
|       Documentation and resources
+---tests
|       test_unredactor.py
```


## Dataset Structure

The project uses the following datasets:
1. **`unredactor.tsv`**: Contains training and validation data.
   - Columns:
     - `split`: Indicates whether the row belongs to training or validation (`training`/`validation`).
     - `name`: The redacted name.
     - `context`: Text context containing the redaction block (`█`).

2. **`test.tsv`**: Contains test data with columns:
   - `id`: Unique identifier for each row.
   - `context`: Text context containing the redaction block.

Output predictions for the test set are saved in `submission.tsv`.

## Code Description

### Key Functions

#### `clean_and_tokenize(text)`
Cleans the text and tokenizes it using NLTK's tokenizer. Special handling is included for the redaction block (`█REDACTED█`).

#### `extract_features(row)`
Extracts a feature dictionary from each row of the dataset. Features include:
- Words and bigrams surrounding the redaction block.
- Length of the redacted block.
- Sentiment score of the context.
- Named entity recognition (NER) results for detecting the presence of personal names.

#### `train_model(X_train, y_train)`
Trains a machine learning pipeline (Random Forest classifier) using extracted features.

#### `evaluate_model(y_true, y_pred)`
Calculates and prints evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score




## Evaluation Metrics

On the validation dataset, the model outputs:
- **Accuracy**: Measures overall correctness.
- **Precision**: Indicates the proportion of correct positive predictions.
- **Recall**: Measures the ability to identify all relevant instances.
- **F1-Score**: Harmonic mean of precision and recall.

Example output:
```
Validation Metrics:
Accuracy: 0.85
Precision: 0.88
Recall: 0.84
F1-Score: 0.86
```


## Assumptions and Limitations

- The redaction block is assumed to replace personal names only.
- SpaCy's NER model may not always identify names accurately.
- The extracted features may not capture all nuances, leading to some mispredictions.
- Limited dataset size restricts model generalization.

## Tests

Unit tests are provided in `tests/test_unredactor.py`.



