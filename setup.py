from setuptools import setup, find_packages

setup(
    name="credit_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit==1.31.0',
        'numpy==1.24.3',
        'pandas==2.0.2',
        'scikit-learn==1.2.2',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'imbalanced-learn==0.11.0',
        'xgboost==2.0.3',
    ],
)
