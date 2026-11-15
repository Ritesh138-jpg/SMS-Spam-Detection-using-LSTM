# SMS Spam Detection Using Long Short-Term Memory (LSTM) Networks

An academic NLP project that classifies SMS messages as **spam** or **ham** (legitimate) using a Keras LSTM network.

## Project status
Starter implementation. The model and metrics are produced when you train it; no accuracy result is claimed in advance.

## Dataset
Use the UCI SMS Spam Collection:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Download the dataset, extract the file named `SMSSpamCollection`, and place it at `data/SMSSpamCollection`.

Labels: `ham` = legitimate (0), `spam` = spam (1).

## Setup
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python train.py
```

The script loads and validates data, creates stratified train/validation/test splits, learns the tokenizer from training data only, trains an Embedding + LSTM classifier, reports accuracy/precision/recall/F1 and a confusion matrix, and saves the model and tokenizer under `models/`.

## Run the demo

```bash
streamlit run app.py
```

Enter an SMS message to receive a spam/ham prediction and estimated spam probability.

## Structure

```text
SMS-Spam-Detection-using-LSTM/
├── data/                  # Put UCI SMSSpamCollection here (not committed)
├── models/                # Generated model and tokenizer (not committed)
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## Reporting notes
- Report the metrics produced by your own training run; do not invent or copy results.
- Discuss class imbalance and include precision, recall, F1 and a confusion matrix, not accuracy alone.
- Cite the UCI dataset and any papers or external code consulted.
- The implementation is original starter code; cite external references you use.
