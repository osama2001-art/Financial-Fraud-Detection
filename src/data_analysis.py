import pandas as pd
from pathlib import Path


def load_data():
    """Load the PaySim dataset."""

    project_folder = Path(__file__).resolve().parents[1]
    file_path = project_folder / "data" / "paysim dataset.csv"

    data = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Total transactions:", len(data))

    return data


def explore_data(data):
    """Explore the dataset."""

    print("\nFirst five transactions:")
    print(data.head())

    print("\nDataset dimensions:")
    print(data.shape)

    print("\nFraud distribution:")
    print(data["isFraud"].value_counts())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nTransaction types:")
    print(data["type"].value_counts())