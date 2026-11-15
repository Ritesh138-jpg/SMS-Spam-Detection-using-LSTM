# Project Documentation
## SMS Spam Detection Using Long Short-Term Memory (LSTM) Networks

**Model:** LSTM | **Language:** Python | **Framework:** TensorFlow/Keras | **Interface:** Streamlit
**Dataset:** UCI SMS Spam Collection

> This is a concise documentation draft. Add team names, guide details, screenshots, literature survey, diagrams and institutional front matter before formal submission. Do not describe unperformed experiments as completed.

## 1. Abstract
This project develops a text classification system that predicts whether an SMS message is spam or legitimate (ham). It uses the UCI SMS Spam Collection, tokenization, padded word sequences and an LSTM neural network implemented with TensorFlow/Keras. A Streamlit interface accepts a message and displays the predicted class and a model confidence estimate. In the recorded experiment, the model achieved 99.03% accuracy, 98.36% spam precision, 93.75% spam recall and 96.00% spam F1-score on a held-out test set of 516 messages. These figures describe the recorded split, not guaranteed real-world performance.

## 2. Problem Statement
Manual identification of spam SMS messages is difficult because messages use varied wording, abbreviations, promotional language and deceptive claims. This project learns patterns from labelled examples and classifies an input message as spam or ham.

## 3. Objectives
- Prepare a labelled SMS dataset for sequence modelling.
- Implement an LSTM-based binary text classifier.
- Evaluate using accuracy, precision, recall, F1-score and a confusion matrix.
- Build an interface for testing new messages.
- Document model performance and limitations.

## 4. Dataset
UCI SMS Spam Collection v.1 contains 5,574 messages: 4,827 ham (86.6%) and 747 spam (13.4%). Labels are mapped to ham = 0 and spam = 1. The corpus is imbalanced, so accuracy must be considered alongside class-specific precision, recall and F1. The raw dataset is downloaded separately and is not committed to this repository.

## 5. Methodology
1. Load the tab-separated corpus and validate labels and message text.
2. Remove empty records and duplicate message texts.
3. Split into 80% training, 10% validation and 10% test data using stratification.
4. Fit the tokenizer on training text only.
5. Convert text to integer sequences and pad/truncate to 100 tokens.
6. Train an Embedding + LSTM + Dropout + Dense classifier.
7. Use Adam, binary cross-entropy and early stopping on validation loss.
8. Predict on the held-out test set with a 0.5 threshold.
9. Calculate metrics and save the model, tokenizer and metrics JSON.
10. Load saved artifacts in Streamlit for interactive prediction.

## 6. Model Architecture
- Input: padded integer sequence of length 100.
- Embedding: 64-dimensional learned word vectors.
- LSTM: 64 recurrent units with dropout 0.2.
- Dropout: 0.3.
- Dense hidden layer: 32 units, ReLU.
- Output: one sigmoid unit representing estimated spam probability.
A sigmoid output greater than or equal to 0.5 is classified as spam; otherwise it is ham.

## 7. Experimental Setup
- Random seed: 42.
- Vocabulary limit: 10,000.
- Maximum sequence length: 100.
- Embedding dimension: 64.
- Batch size: 32.
- Maximum epochs: 15.
- Early stopping: validation loss, patience 3, restore best weights.
- Recorded held-out test set: 516 messages.

## 8. Recorded Results
| Metric | Result |
|---|---:|
| Accuracy | 99.03% |
| Spam precision | 98.36% |
| Spam recall | 93.75% |
| Spam F1-score | 96.00% |
| Test samples | 516 |

Confusion matrix (rows = actual class; columns = predicted class):

| | Predicted ham | Predicted spam |
|---|---:|---:|
| Actual ham | 451 | 1 |
| Actual spam | 4 | 60 |

The model correctly classified 451 ham and 60 spam messages. One ham message was incorrectly flagged as spam, while four spam messages were missed. Spam recall measures the proportion of actual spam messages detected.

## 9. Application Features
- SMS text input and spam/ham prediction.
- Confidence estimate for the predicted class.
- Recent prediction history for the current session.
- Dataset class counts.
- Display of saved test metrics and confusion matrix.

## 10. Limitations
- The dataset may not represent current spam styles, languages or regions.
- The dataset is imbalanced.
- A high score on one split does not guarantee generalization.
- Displayed confidence is not calibrated certainty.
- The model examines text only; it does not verify sender identity, URLs or linked websites.
- This is an academic prototype, not a production security filter.

## 11. Future Enhancements
- Evaluate on an independent SMS dataset.
- Tune the classification threshold using validation data.
- Compare with TF-IDF + Logistic Regression.
- Analyse false positives and false negatives.
- Explore multilingual SMS classification.
- Add safe URL feature extraction without opening suspicious links.

## 12. Suggested Team Work Distribution
- Member 1: dataset understanding, preprocessing and literature survey.
- Member 2: LSTM architecture, training and evaluation.
- Member 3: Streamlit interface, integration, testing and report formatting.
All members should understand the full pipeline.

## 13. References
1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, T. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering.
2. UCI Machine Learning Repository. SMS Spam Collection. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
3. TensorFlow. Keras API documentation. https://www.tensorflow.org/api_docs/python/tf/keras
4. TensorFlow. Text classification tutorials. https://www.tensorflow.org/tutorials/keras/text_classification
5. Scikit-learn. Model evaluation documentation. https://scikit-learn.org/stable/modules/model_evaluation.html
6. Streamlit. Documentation. https://docs.streamlit.io/

Add further verified research papers and sources to meet your institution’s required reference count. Verify bibliographic details before submission.