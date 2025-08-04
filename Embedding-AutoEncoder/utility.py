import matplotlib.pyplot as plt
import subprocess

def plot_loss_curve(train_losses, test_losses):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss", marker='o')
    ax.plot(test_losses, label="Test Loss", marker='x')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Test Loss")
    ax.legend()
    ax.grid(True)
    return fig


def get_gpu_utilization():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
         '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE,
        text=True
    )
    gpu_util, mem_used = map(int, result.stdout.strip().split(','))
    return gpu_util, mem_used

