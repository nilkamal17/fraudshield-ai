# FraudShield AI — Real-Time Financial Fraud Detection

FraudShield AI is an end-to-end **student portfolio project** that demonstrates how a machine-learning model can screen financial transactions and flag suspicious activity for human review.

> **Important:** This is an educational prototype. The included demo data is generated and does not represent real bank transactions. The dashboard provides local, on-demand scoring; it is not connected to a live bank, payment gateway, Kafka stream, or production fraud operations system. Never block a real customer or accuse a person based only on this model.

## 1. Problem statement

Financial services need to review large numbers of transactions, while fraudulent transactions are relatively rare. This project demonstrates a machine-learning workflow that estimates transaction fraud risk and helps an analyst prioritise transactions for review.

## 2. What you will build

- A reproducible transaction dataset for local practice.
- A data validation and feature-engineering pipeline.
- A class-weighted Random Forest classifier.
- A validation-based decision threshold that favours fraud recall (F2 score).
- Evaluation using Precision, Recall, F1, PR-AUC, ROC-AUC, and a confusion matrix.
- A Streamlit dashboard for single-transaction and CSV batch screening.
- Automated tests with Pytest.

## 3. Technology stack

| Area | Tools |
|---|---|
| Language | Python 3.11 (3.10+ should also work) |
| Data work | Pandas, NumPy |
| Machine learning | Scikit-learn, Random Forest |
| Model storage | Joblib |
| Dashboard | Streamlit |
| Testing | Pytest |
| Version control | Git and GitHub |

## 4. Project architecture

```text
FraudShield_AI/
│
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── README.md                      # Project tutorial
├── .gitignore                     # Excludes environment and local artifacts
│
├── data/
│   └── demo_transactions.csv      # Included synthetic demo data
│
├── src/
│   ├── __init__.py
│   ├── features.py                 # Validation and feature engineering
│   ├── model.py                    # Scikit-learn pipeline
│   ├── inference.py                # Model loading and fraud scoring
│   └── train.py                    # Split, train, tune, evaluate, save
│
├── scripts/
│   ├── __init__.py
│   └── generate_demo_data.py       # Rebuild the synthetic demo data
│
├── models/
│   └── .gitkeep                    # Trained model is generated locally
│
├── reports/
│   └── .gitkeep                    # Evaluation report is generated locally
│
└── tests/
    ├── test_features.py
    └── test_model_and_scoring.py
```

## 5. Open the project in VS Code (Windows)

1. Download `FraudShield_AI.zip`.
2. In File Explorer, right-click the ZIP and choose **Extract All**.
3. Open **Visual Studio Code**.
4. Choose **File → Open Folder**.
5. Select the extracted `FraudShield_AI` folder — select the folder that directly contains `app.py`, `requirements.txt`, `src`, and `data`.
6. Choose **Terminal → New Terminal**.

### Check that the terminal is in the project root

The commands below are written for **Command Prompt (CMD)**, which is the terminal you used previously. Run:

```cmd
cd
```

Then:

```cmd
dir
```

You should see `app.py`, `requirements.txt`, `README.md`, `src`, `scripts`, `data`, and `tests`. If you do not see them, open the correct extracted folder in VS Code before continuing.

## 6. Create and activate a virtual environment

Keep the terminal in the project root.

```cmd
py --version
```

Create the environment:

```cmd
py -3.11 -m venv .venv
```

If your computer does not have Python 3.11 installed, use the Python version you have:

```cmd
python -m venv .venv
```

Activate it in **CMD**:

```cmd
.venv\Scripts\activate.bat
```

When active, the terminal normally begins with `(.venv)`.

Upgrade pip and install the libraries:

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If VS Code asks you to select a Python interpreter, press `Ctrl + Shift + P`, choose **Python: Select Interpreter**, and select the interpreter inside `.venv`.

## 7. Train your first model

The project already includes `data/demo_transactions.csv`. You do not need to download anything to test the workflow.

From the project root, run:

```cmd
python -m src.train
```

The script will:

1. Load and validate the transaction data.
2. Create amount and time features.
3. Split the data into training, validation, and test sets using stratification.
4. Train a class-weighted Random Forest.
5. Select a decision threshold on validation data using an F2 objective.
6. Evaluate once on the held-out test set.
7. Save the model and evaluation report.

Generated files:

```text
models/fraudshield_model.joblib
reports/metrics.json
```

The displayed metrics from the included generated data are only a software-pipeline demonstration. They are not evidence of performance on real financial transactions.

## 8. Run the dashboard

After training completes, run:

```cmd
streamlit run app.py
```

If the `streamlit` command is not recognised, use:

```cmd
python -m streamlit run app.py
```

Streamlit will print a local address, normally `http://localhost:8501`. Open that address in your browser.

### Dashboard pages

1. **Overview** — project summary and model metrics.
2. **Single Transaction Check** — enter a transaction type, amount, and time step to receive a risk score.
3. **Batch CSV Screening** — upload a CSV, score all rows, and download the results.
4. **Model Performance** — precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, and feature importance.
5. **About & Limitations** — dataset and responsible-use notes.

## 9. Run automated tests

Stop the dashboard with `Ctrl + C`, then run:

```cmd
pytest -q
```

The tests check the data validation, feature creation, model fitting, and scoring pipeline.

## 10. Optional: train with the public PaySim dataset

For a stronger portfolio experiment, you can download the public **Synthetic Financial Datasets for Fraud Detection (PaySim)** CSV from Kaggle:

<https://www.kaggle.com/datasets/ealaxi/paysim1>

You need a Kaggle account to download it. Extract the CSV and put it inside this project's `data` folder. The usual filename is:

```text
data/PS_20174392719_1491204439457_log.csv
```

The model expects these columns from PaySim:

- `step`
- `type`
- `amount`
- `isFraud`

The script ignores additional columns. To keep training manageable on a student laptop, use a stratified sample of 250,000 rows:

```cmd
python -m src.train --data data\PS_20174392719_1491204439457_log.csv --sample-size 250000
```

Then restart the dashboard:

```cmd
streamlit run app.py
```

**Feature-selection note:** The PaySim dataset publisher specifically cautions against using its origin/destination balance fields for fraud detection because simulated fraudulent transactions are cancelled. This project therefore does not use `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, or `newbalanceDest` as model inputs. Read the dataset description and cite the dataset if you use it in your report: <https://www.kaggle.com/datasets/ealaxi/paysim1>.

PaySim is a synthetic simulation of mobile-money activity, not a dump of actual customer transactions. A published description reports 6,362,620 simulated transactions and 8,213 fraud examples; the class imbalance is one reason to report Precision, Recall and PR-AUC rather than accuracy alone. See the dataset source and overview paper:

- Dataset: <https://www.kaggle.com/datasets/ealaxi/paysim1>
- Lopez-Rojas et al. (2016), *PaySim: A financial mobile money simulator for fraud detection*.

Do not upload the full PaySim CSV to GitHub. `.gitignore` excludes downloaded data CSVs other than the small included demo file.

## 11. CSV format for batch prediction

The uploaded CSV must contain at least these three columns (column names are case-sensitive):

```csv
step,type,amount
12,TRANSFER,15000.00
35,PAYMENT,250.00
48,CASH_OUT,80000.00
```

An optional `isFraud` column is ignored during prediction. Do not include card numbers, account numbers, names, phone numbers, addresses, or other personal information in public examples or GitHub commits.

## 12. Push the project to GitHub

First create an **empty public repository** named `fraudshield-ai` on GitHub. Do not initialise it with a README, `.gitignore`, or license because this project already contains those files.

In the VS Code terminal, from the project root:

```cmd
git init
git add .
git commit -m "Initial commit - FraudShield AI"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/fraudshield-ai.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username. If Git says `origin` already exists, update it instead of adding it again:

```cmd
git remote set-url origin https://github.com/YOUR_GITHUB_USERNAME/fraudshield-ai.git
```

The `.venv`, local trained model, generated metrics report, and downloaded PaySim data are excluded from Git. After cloning the repository on another computer, install the requirements and run `python -m src.train` to create a local model and report.

## 13. Troubleshooting

### `requirements.txt` not found

Your terminal is not in the folder containing `requirements.txt`. Run `dir`. If you cannot see `app.py` and `requirements.txt`, use **File → Open Folder** and select the extracted project root.

### `No module named src`

Run the module command from the project root:

```cmd
python -m src.train
```

Do not run `python src\train.py` from another folder.

### Model file not found

Train before starting the dashboard:

```cmd
python -m src.train
```

### `streamlit` not recognised

```cmd
python -m streamlit run app.py
```

### Deactivate the virtual environment

When you are finished, run:

```cmd
deactivate
```

You can keep the virtual environment active while working on this project.

## 14. Suggested future improvements

- Compare Random Forest with Logistic Regression and HistGradientBoosting.
- Add time-aware chronological validation instead of random splitting.
- Add transaction-history features such as recent transaction count and deviation from a customer's normal amount (only where valid historical data is available).
- Add model calibration, SHAP explanations, and drift monitoring.
- Build a FastAPI endpoint and connect an event-stream simulator for a more realistic streaming architecture.
- Validate the system with a bank-approved, privacy-safe dataset before making any operational claims.

## 15. Resume description (edit this after you run and understand it)

**FraudShield AI — Financial Transaction Fraud Screening:** Built an end-to-end Python machine-learning prototype using Pandas and Scikit-learn to screen transaction records, address class imbalance with class-weighted Random Forest, tune a validation decision threshold, evaluate with Precision/Recall/F1/PR-AUC, and deliver single-record and batch predictions through a Streamlit dashboard.
"# fraudshield-ai" 
