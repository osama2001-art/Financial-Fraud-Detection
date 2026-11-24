Financial Fraud Detection & Transaction Risk Intelligence System

A machine learning project that analyzes completed financial transactions, identifies potentially fraudulent activity, and assigns risk levels to suspicious transactions.

Project Overview

Financial fraud detection is an important challenge for financial institutions. This project explores how machine learning can help identify suspicious transactions and support post-transaction fraud monitoring.

The system uses the PaySim synthetic financial transaction dataset to train and evaluate three classification models. It also generates transaction risk scores and provides an interactive dashboard for reviewing flagged transactions.

Project type: Machine Learning / Financial Technology
Language: Python
Framework: Streamlit
Libraries: Pandas, NumPy, Scikit-learn, Plotly

Key Features

* Financial transaction data analysis
* Feature engineering
* Chronological training and testing split
* Three machine learning classification models
* Fraud probability and risk scoring
* HIGH and MEDIUM risk alerts
* CSV export of flagged transactions
* Interactive Streamlit dashboard
* Model performance comparison

Machine Learning Models

The project implements and compares:

1. Decision Tree
2. Random Forest
3. Logistic Regression

The models are evaluated using precision, recall, F1-score, and accuracy.

Model Performance

Results on the chronological test dataset:

Model	Precision	Recall	F1 Score
Decision Tree	98.77%	92.44%	95.50%
Random Forest	99.27%	82.59%	90.17%
Logistic Regression	98.88%	96.13%	97.49%

Logistic Regression achieved the highest F1-score and recall among the three models in this experiment.

Fraud Risk Scoring

The Logistic Regression model generates a fraud probability score for each transaction.

Transactions are assigned risk levels using the following experimental thresholds:

* HIGH: Risk score ≥ 0.80
* MEDIUM: Risk score ≥ 0.40 and below 0.80
* LOW: Risk score below 0.40

The dashboard focuses on flagged HIGH and MEDIUM risk transactions.

In the evaluated test results:

* 1,621 transactions were flagged.
* 1,594 were classified as HIGH risk.
* 27 were classified as MEDIUM risk.
* 1,600 flagged transactions had an actual fraud label in the dataset.

These are experimental results on the selected test dataset, not real-world banking performance.

Dataset

This project uses the PaySim synthetic financial transaction dataset.

The dataset contains simulated mobile-money transactions and includes transaction types, amounts, account balances, and fraud labels.

The dataset contains approximately 6.36 million transactions, with a very small proportion labeled as fraudulent.

The original dataset is not included in this repository. Place the downloaded CSV file in the data/ directory using the expected filename:

paysim dataset.csv

Project Structure

Financial_Fraud_Detection/
│
├── data/
│   └── paysim dataset.csv
│
├── src/
│   ├── data_analysis.py
│   ├── main.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── dashboard.py
│   └── fraud_alerts.csv
│
├── README.md
└── requirements.txt

Installation and Usage

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd Financial_Fraud_Detection

2. Install dependencies

pip install -r requirements.txt

3. Add the dataset

Place paysim dataset.csv inside the data/ folder.

4. Run the machine learning pipeline

python src/main.py

5. Launch the dashboard

python -m streamlit run src/dashboard.py

Methodology

1. Load and explore the transaction dataset.
2. Analyze fraud distribution and transaction characteristics.
3. Engineer additional transaction features.
4. Split the data chronologically to evaluate on later transactions.
5. Train and compare three classification models.
6. Evaluate model performance using classification metrics.
7. Generate fraud probability scores and risk categories.
8. Export flagged transactions.
9. Visualize results through an interactive dashboard.

Limitations and Future Considerations

* PaySim is a synthetic dataset and does not fully represent real banking transactions.
* Strong performance on this dataset does not guarantee performance on real financial data.
* Risk thresholds are experimental and would require further validation before practical deployment.
* Further testing on different datasets and changing fraud patterns would be necessary.
* The current project is a prototype, not a production banking fraud detection system.

Author

Osama Almesaimi

Master’s Student in Artificial Intelligence
University of Massachusetts Lowell
