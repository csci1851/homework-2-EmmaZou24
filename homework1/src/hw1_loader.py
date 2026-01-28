"""Data loader for Homework 1: Regression and Classification"""

import os
import pandas as pd
from typing import Tuple


class HW1DataLoader:
    def get_aging_data(self, csv_path=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the genomic aging dataset from CSV file.
        Returns:
            X: Features DataFrame containing all the CpG data
            y: Target Series (age)
            metadata: DataFrame containing metadata
        """
        try:
            df = pd.read_csv(csv_path)
            X = df.iloc[:, 4:]
            y = df.iloc[:, 2]
            metadata = df.iloc[:, [0, 1, 3]]
            return X, y, metadata
        except Exception as e:
            print(f"Error loading genomic aging dataset: {e}")
            return None

    def get_heart_disease_data(self, csv_path=None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the Heart Disease dataset from CSV file.
        Returns:
            X: Features DataFrame
            y: Target Series (presence of heart disease)
        """
        try:
            data = pd.read_csv(csv_path)
            print(f"Successfully loaded heart disease data with {len(data)} rows")

            target_col = "target"
            X = data.drop(target_col, axis=1)
            y = pd.Series(data[target_col], name=target_col)

            return X, y
        except Exception as e:
            print(f"Error loading heart disease data: {e}")
            return None, None
