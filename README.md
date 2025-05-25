# Dual-Disease-Risk-Prediction-Tool
## 📋 Project Overview
Machine learning system that predicts diabetes and heart disease risk with:
- 74% prediction accuracy 
- 82% sensitivity (recall)
- 30% improved interpretability for clinicians

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
git clone https://github.com/yourusername/disease-risk-prediction.git
cd disease-risk-prediction
pip install -r requirements.txt

python models/train_model.py --data=data/health_data.csv

python visualization/risk_factors.py

jupyter notebook visualization/dashboard.ipynb

.
├── data/                   # Health datasets
│   ├── health_data.csv     # Sample dataset (5000 records)
│   └── preprocessing.py    # Data cleaning pipeline
├── models/
│   ├── train_model.py      # Model training script
│   └── disease_model.pkl   # Pretrained model
├── visualization/
│   ├── risk_factors.py     # Visualization generator
│   ├── dashboard.ipynb     # Interactive dashboard
│   └── risk_factors.png    # Sample output
├── requirements.txt        # Dependencies
└── README.md

