#!/usr/bin/env python
"""Setup script for Garment Productivity Prediction package"""

from setuptools import setup, find_packages

setup(
    name="garment-productivity-prediction",
    version="1.0.0",
    description="ML Pipeline for Garment Productivity Prediction with MLOps",
    author="Fares",
    author_email="fares@example.com",
    url="https://github.com/fares279/garment-productivity-prediction",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "mlflow>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "monitoring": [
            "elasticsearch==8.11.0",
            "psutil>=5.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
