# CIS6930FA24 -- Project 2: The Unredactor

**Name:** Prajay Yalamanchili

## Project Description

This project is focused on building an Unredactor, a machine learning model designed to predict the most likely names to replace redacted information from text documents. The project involves:

- Training a machine learning pipeline to predict names based on contextual clues.
- Evaluating the model's performance on a validation set using metrics such as accuracy, precision, recall, and F1-score.
- Generating predictions for a provided test dataset and saving them in a specified format.

The redacted content in the dataset is represented by sequence of block characters (`█`). The task is to accurately unredact these blocks, focusing primarily on personal names.

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

To execute the `unredactor.py` file, use:
```bas
pipenv run python unredactor.py 
```
To run tests, use:
```bash
pipenv run python -m pytest -v
```
or

```bash
pipenv run pytest
```

## Demo Program Execution


[watch](https://github.com/user-attachments/assets/ae891fc9-d8a5-4708-b190-472f0499b7d7)

## Folder Structure

```
|   README.md
|   unredactor.py
|   unredactor.tsv
|   test.tsv
|   submission.tsv
|   COLLABORATORS.md
|   LICENSE
|
+---tests
|       test_unredactor.py
```
- **COLLABORATORS.md:** Contains information about collaborators and a list of resources used for the assignment.
- **unredactor.py:** The main Python script containing the implementation of the unredactor pipeline, including data reading, feature extraction, model training, evaluation, and test prediction.
- **Pipfile:** Manages the Python virtual environment and lists all dependencies.
- **Pipfile.lock:** Specifies the versions of dependencies to ensure consistent environments.
- **README.md:** This file, which documents the assignment.
- **LICENSE:** Contains licensing information, including copyright, publishing, and usage rights.
- **tests:** Contains test files. `test_unredactor.py` is used for testing the unredactor Python file.



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

## Pipeline

1. **Data Reading and Preprocessing**:
   - Reads `unredactor.tsv` and splits it into training and validation sets.
   - Cleans and tokenizes the text data.

2. **Feature Extraction**:
   - Extracts contextual features, including:
     - Previous and next words around the redaction.
     - Part-of-speech (POS) tags.
     - TF-IDF scores.
     - Sentiment analysis scores.
     - Named entity recognition (NER) features.
   - Converts these features into numerical vectors using `DictVectorizer`.

3. **Model Training**:
   - Trains a `RandomForestClassifier` using the training dataset.
   - Evaluates the model on the validation set using metrics such as precision, recall, F1-score, and accuracy.

4. **Prediction**:
   - Uses the trained model to predict redacted names in `test.tsv`.
   - Outputs predictions in `submission.tsv` for evaluation or further use.


## `unredactor.py`

**Functions in `unredactor.py`:**
### `main()`
The central function that orchestrates the entire unredaction pipeline. It reads the input data, extracts features, trains the machine learning model, evaluates its performance on validation data, and predicts unredacted names from the test dataset.

### `read_data()`
Reads the `unredactor.tsv` file, preprocesses the text data by filtering valid rows, splits the data into training and validation sets based on the split column, and fits the TfidfVectorizer on the context column of the training dataset.

### `feature_extraction(train_data, val_data)`
Extracts meaningful features for training and validation datasets by processing each row using `extract_features`. Converts the extracted feature dictionaries into numerical feature vectors using a DictVectorizer for compatibility with machine learning models. Returns the feature vectors for both datasets and the trained DictVectorizer.

### `extract_features(row)`
Generates a dictionary of features from a single row of the dataset by:
1. Extracting **contextual information** such as three preceding and succeeding words (`prev_three_words`, `next_three_words`) along with their POS tags (`prev_three_pos`, `next_three_pos`).
2. Calculating **TF-IDF scores** (`prev_tfidf_mean`, `next_tfidf_mean`) for the contextual words.
3. Computing the **redaction metadata** such as the length of the redacted text (`redaction_length`).
4. Adding **sentiment analysis** scores for the context (`sentiment_compound`).
5. Counting the number of person entities detected using named entity recognition (`person_entities_count`).

### `train(X_train, y_train, X_val, y_val)`
Trains a `RandomForestClassifier` pipeline using the training features and labels (X_train, y_train), evaluates it on the validation dataset (X_val, y_val), and prints key performance metrics including accuracy, precision, recall, and F1-score. Returns the trained pipeline.

### `predict_test_tsv(pipeline, vec)`
Reads test.tsv to process the test dataset, extracts features for each row using `extract_features`, and transforms them into feature vectors using the trained DictVectorizer. Uses the trained pipeline to predict redacted names and saves the results (id, name) into submission.tsv.

### `evaluation(y_val, y_pred)`
The `evaluation` function computes key performance metrics, including accuracy, precision, recall, and F1-score, to assess the model's performance on validation data. It uses y_va` (true labels) and y_pred (predicted labels) as inputs and outputs the metrics in a formatted console-friendly summary. This function provides a quick and comprehensive evaluation of the model's effectiveness.  


## `test_unredactor.py`

**Functions in `test_unredactor.py`:**

### `test_clean_and_tokenize()`
This test function verifies the correctness of the `clean_and_tokenize()` function, which is responsible for cleaning and tokenizing input text.



## Evaluation Metrics

On the validation dataset, the model outputs:
- **Accuracy**: Measures overall correctness.
- **Precision**: Indicates the proportion of correct positive predictions.
- **Recall**: Measures the ability to identify all relevant instances.
- **F1-Score**: Harmonic mean of precision and recall.

Example output:
```
Validation Metrics:
Accuracy: 0.07
Precision: 0.05
Recall: 0.05
F1-Score: 0.0.04
```


## Assumptions and Limitations

- The accuracy of model is low because of limited training dataset.
- The extracted features may not capture all nuances, leading to some mispredictions.
- The redaction block is assumed to replace personal names only.
- SpaCy's NER model may not always identify names accurately.




