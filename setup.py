from setuptools import setup, find_packages

setup(
    name="netguard-ai",
    version="0.1.0",
    description="An AI system that predicts network cyberattacks BEFORE they complete using World Models.",
    author="Your Name/Team",
    url="https://github.com/yourusername/ai-network-attack-forecaster",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "joblib==1.4.2",
        "matplotlib==3.9.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "plotly==5.24.1",
        "pyshark==0.6.0",
        "scapy==2.6.1",
        "scikit-learn==1.5.2",
        "seaborn==0.13.2",
        "shap==0.46.0",
        "streamlit==1.40.0",
        "torch==2.5.1",
        "tqdm==4.67.0",
    ],
    entry_points={
        "console_scripts": [
            "netguard-predict=predict:main",
        ],
    },
)
