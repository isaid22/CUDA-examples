#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ModelStructure import TitanicAutoencoder, TitanicAutoencoderDataset
from utility import plot_loss_curve, get_gpu_utilization
import LoadData

def train_model(X_train, X_test, train_ds, test_ds, latent_dim, device, config):


        
    batch_size = config.get('training', {}).get('batch_size')
    LATENT_DIM = latent_dim  # Set the latent dimension for the autoencode
    print(f"Using input dimension: {X_train.shape[1]}, latent dimension: {LATENT_DIM}")
    model = TitanicAutoencoder(input_dim=X_train.shape[1], latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5) 
    # This will halve the learning rate if avg_test_loss doesn't improve for 5 epochs.

    loss_fn = nn.MSELoss()

    starting_epoch = 0 # Define starting epoch  0

    # Track losses
    train_losses = []
    test_losses = []

    if config.get('training', {}).get('resume_training_from_checkpoint'):
        # Load the checkpoint
        starting_epoch, train_losses, test_losses = LoadData.load_checkpoint(
        config, model, optimizer, scheduler, device
    )


    # Create a timestamped run directory
    os.makedirs("runs", exist_ok=True) # Create directories for logs and checkpoints
    RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_DIR = f"runs/autoencoder_{RUN_NAME}" # Create directories for logs and checkpoints
    CHECKPOINT_DIR = f"checkpoints_{RUN_NAME}" # Create directories for logs and checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Create directories if they don't exist
    BATCH_SIZE = config.get('training', {}).get('batch_size', 256) # Adjust batch size as needed default 256
    WORKERS = config.get('training', {}).get('num_workers', 1) # Number of workers for DataLoader, adjust based on your system

    writer = SummaryWriter(log_dir=LOG_DIR)

    #Training loop
    # Prepare training and test datasets
    train_ds = TitanicAutoencoderDataset(X_train)
    test_ds = TitanicAutoencoderDataset(X_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=WORKERS)


    best_loss = float('inf') # Initialize best_loss for comparison and checkpointing
    # Training loop
    
    num_epochs = config.get('training', {}).get('num_epochs', 100) # Get number of epochs from config default 100
    for epoch in range(starting_epoch, starting_epoch + num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            recon = model(x)
            loss = loss_fn(recon, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        end_time = time.time()
        epoch_duration = end_time - start_time
        # GPU metrics
        gpu_alloc = torch.cuda.memory_allocated() / 1024**2 # Convert to MB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**2 # Convert to MB
        writer.add_scalar("GPU/Memory_Allocated_MB", gpu_alloc, epoch)
        writer.add_scalar("GPU/Memory_Reserved_MB", gpu_reserved, epoch)

        try:
            gpu_util, mem_used = get_gpu_utilization()
            writer.add_scalar("GPU/Utilization_%", gpu_util, epoch)
            writer.add_scalar("GPU/Memory_Used_MB", mem_used, epoch)
        except:
            pass  # In case nvidia-smi isn't available

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                recon = model(x)
                loss = loss_fn(recon, y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # ✅ Log scalar losses
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/test", avg_test_loss, epoch)
        writer.add_figure("Loss Overlap Curve", plot_loss_curve(train_losses, test_losses), global_step=epoch)

        # ✅ Log weights and gradients
        for name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        # ✅ Log activation from encoder
        with torch.no_grad():
            activation_sample = torch.tensor(X_train[:1], dtype=torch.float32).to(device)
            encoded = model.encoder(activation_sample)
            writer.add_histogram("Activations/EncoderOutput", encoded, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR", current_lr, epoch)

        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_duration:.2f} sec")

        # Save only the best model based on test loss
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            print(f"New best model found at epoch {epoch+1}, saving checkpoint...")
            
                # Save the model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'input_dim': X_train.shape[1],           # ✅ Save architecture args
                'latent_dim': LATENT_DIM 
                }, f"{CHECKPOINT_DIR}/autoencoder_epoch{epoch+1}.pt")
        
        scheduler.step(avg_test_loss) # Adjust learning rate based on test loss

    # ✅ After training — log embeddings to projector
    with torch.no_grad():
        sample_input = torch.tensor(X_test[:500], dtype=torch.float32).to(device)
        latent_vectors = model.encoder(sample_input)
        metadata = [f"Passenger {i}" for i in range(sample_input.shape[0])]
        writer.add_embedding(latent_vectors, metadata=metadata, tag="LatentEmbeddings", global_step=num_epochs)
        
    writer.close()

