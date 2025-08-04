import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import yaml
import os 



def load_and_split_data(data_frame, train_fraction=0.8, test_fraction=0.2, seed_num=42):
    """
    Load data and split into training, test, and seed for reproducibility.
    """
    FRACTION = train_fraction

    ## Drop columns with missing value and alive column
    df_nomissing = data_frame.drop(columns=['age', 'deck', 'embarked', 'embark_town', 'survived',])
    # Normalize numeric feature

    num_cols = ["fare", "sibsp", "parch"]
    scaler = StandardScaler()
    df_nomissing[num_cols] = scaler.fit_transform(df_nomissing[num_cols])

    #Convert categorial features to integers using Label Encoding
    cat_cols = ["pclass", "sex", "class", "who", "adult_male", "alive", "alone"]  # treat pclass as categorical
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_nomissing[col] = le.fit_transform(df_nomissing[col])
        label_encoders[col] = le

    # Get the first 80% of rows for the training set
    # This is achieved by sampling a fraction of the DataFrame
    train_df_nomissing = df_nomissing.sample(frac=FRACTION, random_state=seed_num) # fix seed for reproducibility
    # Get the remaining 20% of rows for the test set
    # This is achieved by selecting rows whose index is not present in the training set
    test_df_nomissing = df_nomissing.drop(train_df_nomissing.index)
    # Display the shapes of the resulting DataFrames
    print(f"Shape of training DataFrame: {train_df_nomissing.shape}")
    print(f"Shape of test DataFrame: {test_df_nomissing.shape}")

    # You can now save these DataFrames if needed
    train_df_nomissing.to_csv('train_data_nomissing_std.csv', index=False)
    test_df_nomissing.to_csv('test_data_nomissing_std.csv', index=False)

    return train_df_nomissing, test_df_nomissing

def load_checkpoint(config, model, optimizer, scheduler, device):
    path = config.get('checkpoint', {}).get('path')
    if not os.path.exists(path):
        print(f"❌ Checkpoint not found: {path}")
        return 0, [], []

    print(f"🔄 Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0) + 1
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])

    print(f"✅ Resumed from epoch {epoch}")
    return epoch, train_losses, test_losses