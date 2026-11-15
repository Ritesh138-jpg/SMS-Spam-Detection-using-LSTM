# SMS Spam Detection Using Long Short-Term Memory (LSTM) Networks

An academic NLP project that classifies SMS messages as spam or ham using a TensorFlow/Keras LSTM network.

## Features
- UCI SMS Spam Collection dataset (5,574 labelled messages).
- Training-only tokenizer fitting and padded sequences.
- Embedding + LSTM + Dropout + Dense binary classifier.
- Stratified train/validation/test split and early stopping.
- Streamlit dashboard with predictions, confidence estimate, recent history, dataset statistics and saved test metrics.

## Dataset
Download: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Extract SMSSpamCollection into data/SMSSpamCollection. The raw dataset is excluded from Git. See data/DATASET_SOURCE.md.

## Setup (Windows PowerShell)
Python 3.10 or 3.11 recommended.

    python -m venv venv
    venv\\Scripts\\Activate.ps1
    python -m pip install -r requirements.txt

## Train and evaluate
    python train.py

Generated artifacts: models/sms_lstm.keras, models/tokenizer.joblib and models/metrics.json.

Recorded test run (516 held-out messages): accuracy 99.03%, spam precision 98.36%, spam recall 93.75%, spam F1-score 96.00%. These are results from one split, not a guarantee of real-world performance.

## Run the dashboard
    streamlit run app.py
Open the local URL shown by Streamlit (normally http://localhost:8501).

## Documentation
See docs/PROJECT_DOCUMENTATION.md for methodology, architecture, recorded results, limitations and references.

## Limitations
The dataset is imbalanced. Consider precision, recall and F1 alongside accuracy. Model confidence is not calibrated certainty. The classifier uses text only and does not verify sender identity or URLs. It is an academic prototype, not a production spam filter.

## Citation
Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, T. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering. Also cite the UCI dataset webpage.