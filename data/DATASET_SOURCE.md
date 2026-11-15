# Dataset source and attribution

Dataset: SMS Spam Collection v.1

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The supplied archive contains:
- `SMSSpamCollection`: 5,574 tab-separated messages, each labelled `ham` or `spam`.
- `readme`: original corpus description, statistics, provenance and usage notes.

Original corpus statistics in the supplied readme:
- 4,827 ham messages (86.6%)
- 747 spam messages (13.4%)

Research citation:
Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, T. (2011). Contributions to the study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG '11).

The corpus readme requests citation of the paper and dataset webpage when the corpus is used in research. Please retain this attribution in the project report.

## Local setup
Extract `SMSSpamCollection` from the downloaded ZIP and save it as:
`data/SMSSpamCollection`

The dataset itself is excluded from Git commits by `.gitignore`. Download it directly from UCI rather than committing redistributed copies.
