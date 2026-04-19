import pandas as pd
import numpy as np


def load_and_split_dataset(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Shuffle the dataset
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))

    # Split the dataset
    train_set = data[:train_size]
    val_set = data[train_size:train_size + val_size]
    test_set = data[train_size + val_size:]
    
    # Compute mean and std only on training data
    train_mean = train_set.mean()
    train_std = train_set.std()
    
    return train_set, val_set, test_set, train_mean, train_std


# Example usage:
# train, val, test, mean, std = load_and_split_dataset('path_to_your_dataset.csv')
