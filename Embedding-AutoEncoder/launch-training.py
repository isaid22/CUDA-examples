import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import time
from datetime import datetime
import os
import torch.nn as nn
import yaml
import os
import LoadData
import ModelStructure
from TrainingLoop import train_model

def parse_yaml_config(file_path):
    """
    Parses a YAML file and returns its content as a Python dictionary.
    """
    if not os.path.exists(file_path):
        print(f"Error: YAML file not found at '{file_path}'")
        return None
    try:
        with open(file_path, 'r') as file:
            # Use yaml.safe_load for security when loading from untrusted sources
            config_data = yaml.safe_load(file)
        return config_data
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    

def main(input_args):
    seed_number = input_args.get('seed_number', 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = input_args.get('training', {}).get('learning_rate', 0.001)
    optimizer = input_args.get('training', {}).get('optimizer')
    latent_dim = input_args.get('model', {}).get('latent_dim', 4)
    print("Using device:", device)
    print("Learning rate:", learning_rate)
    print("Optimizer:", optimizer)
    # print('Hello, %s!' % input_args.name)
    # print('Your age is: %d' % input_args.age)
    
    # Load and organize data
    df = sns.load_dataset("titanic") # Load the  dataset
    train_df, test_df = LoadData.load_and_split_data(df, 
        train_fraction=input_args.get('data', {}).get('train_fraction', 0.8),
        test_fraction=input_args.get('data', {}).get('test_fraction', 0.2),
        seed_num=seed_number
        )

    X_train = train_df.values.astype("float32")
    X_test = test_df.values.astype("float32")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    train_ds = ModelStructure.TitanicAutoencoderDataset(X_train)
    test_ds = ModelStructure.TitanicAutoencoderDataset(X_test)

    train_model(X_train, X_test, train_ds, test_ds, latent_dim, device, config)

    return 

if __name__ == '__main__':

    config_file = "config.yaml"  # Name of your YAML configuration file

    # Parse the YAML file
    config = parse_yaml_config(config_file)

    if config:
        print("YAML Configuration Loaded Successfully:")
        print(config)

        # Accessing specific values
        input_dim = config.get('model', {}).get('input_dim')
        latent_dim = config.get('model', {}).get('latent_dim')
        batch_size = config.get('training', {}).get('batch_size')
        starting_epoch = config.get('starting_epoch', 0)
        resume_training_from_checkpoint = config.get('training', {}).get('resume_training_from_checkpoint', False)

        if input_dim:
            print(f"\nModel's input dimension: {input_dim}")
        if not resume_training_from_checkpoint:
            print(f"Training from starting epoch: {starting_epoch}")
        else:
            print(f"Resuming training from checkpoint with starting epoch: {starting_epoch}")
 

    main(config)


