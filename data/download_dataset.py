#!/usr/bin/env python3
import os
import kagglehub

def download_dataset():
    """Download heart disease dataset from Kaggle."""
    print("Downloading heart disease dataset...")
    path = kagglehub.dataset_download("oktayrdeki/heart-disease", path="./data")
    print(f"Dataset downloaded to: {path}")
    return path

if __name__ == "__main__":
    download_dataset()
