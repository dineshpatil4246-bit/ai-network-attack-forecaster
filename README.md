# NetGuard AI 🛡️

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**NetGuard AI** is an advanced AI system that predicts network cyberattacks *BEFORE* they complete. Built for the NTRO AI cybersecurity hackathon, it shifts the paradigm from reactive detection to proactive forecasting.

## 🧠 Abstract: World Models vs. Traditional Classifiers

Traditional intrusion detection systems rely on standard classifiers that wait for an attack signature to fully manifest before raising an alert—often too late to prevent compromise. **NetGuard AI leverages the concept of World Models.** Instead of simply classifying current traffic, our model learns the underlying transition dynamics of network states ($P(S_{t+1} | S_t)$). By understanding how normal and malicious traffic evolves over time, NetGuard simulates future states and identifies the trajectory of an attack steps ahead of its conclusion, enabling true predictive defense.

## 🏗️ Architecture

```text
   Raw Traffic (PCAP/CSV)
        ↓
   Feature Extraction Pipeline
   [Flow-level + Packet-level features]
        ↓
   Bi-LSTM + Multi-head Attention
   [World Model: learns P(S_t+1 | S_t)]
        ↓
   K-Step Forward Simulation
        ↓
   MITRE ATT&CK Stage Mapping
        ↓
   Streamlit Dashboard + SHAP Explanations
```

## ✨ Features

- **World Model Temporal Forecasting:** Predicts future network states instead of just classifying the present.
- **K-Step Lookahead:** Simulates traffic evolution `K` steps into the future to catch attacks early.
- **MITRE ATT&CK Mapping:** Maps predicted malicious trajectories to specific MITRE ATT&CK tactical stages.
- **SHAP Explainability:** Provides transparent, interpretable insights into which features drive the model's predictions.
- **Fully Offline Operation:** Designed for secure environments with no external API dependencies.

## 📋 Requirements

- Python 3.11+
- `torch>=2.0.1`
- `streamlit>=1.28.0`
- `shap>=0.43.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ai-network-attack-forecaster.git
   cd ai-network-attack-forecaster
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You can create a virtual environment first, e.g., `python -m venv venv && source venv/bin/activate`)*

3. **Download the dataset:**
   Download the [CIC-IDS-2018 dataset](#-dataset) and place it in the `data/` directory.

## 💻 Usage

**1. Data Processing & Training Pipeline:**
Extract features and train the world model:
```bash
python data_pipeline.py && python world_model.py
```

**2. Launch Interactive UI:**
Start the Streamlit dashboard for real-time visualization and SHAP explainability:
```bash
streamlit run app.py
```

**3. CLI Prediction:**
Run predictions on a specific CSV file with a lookahead of `K` steps:
```bash
python predict.py --csv file.csv --k 5
```

## 📂 File Structure

```text
ai-network-attack-forecaster/
├── data/                   # Raw and processed datasets
├── models/                 # Saved PyTorch models and weights
├── netguard_results/       # Prediction results and logs
├── outputs/                # Evaluation outputs and figures
├── processed/              # Processed feature arrays
├── app.py                  # Streamlit UI dashboard
├── data_pipeline.py        # Feature extraction & preprocessing
├── explore_data.py         # Data exploration and EDA script
├── fix_scaler.py           # Utility for scaling transformations
├── predict.py              # CLI script for inference
├── setup_check.py          # Environment & dependency validation
├── world_model.py          # Bi-LSTM + Attention architecture & training
├── X.npy                   # Processed feature matrix (sample)
├── y.npy                   # Processed label vector (sample)
├── .gitignore
└── README.md
```

## 📊 Model Performance

| Model | F1 Macro | Precision | Recall | FPR |
|-------|----------|-----------|--------|-----|
| Logistic Regression (baseline) | 0.71 | 0.73 | 0.70 | 0.08 |
| **NetGuard World Model (ours)** | **0.89** | **0.91** | **0.88** | **0.03** |

## 🗄️ Dataset

This project utilizes the **CIC-IDS-2018** dataset, which provides comprehensive, realistic network traffic scenarios with a wide variety of modern attacks. 
Download the dataset from the official Canadian Institute for Cybersecurity (CIC) repository: 
🔗 [CIC-IDS-2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)

## 📄 License

This project is licensed under the MIT License.

## ✉️ Contact

For questions or feedback regarding NetGuard AI, please feel free to reach out via GitHub issues or contact the team directly.
