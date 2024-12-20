import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_avg=pd.read_csv("results/62k_20_epochs/metrics/avg_loss.tsv",sep=' ',names=['run_id', 'avg_loss', 'epoch'])

log=False
loss_values = df_avg["avg_loss"]

print(loss_values)
epochs = range(1, len(loss_values) + 1)  # Epoch numbers
title="Average Loss for 62K batches per epoch"
plt.figure(figsize=(8, 6))
if log:
    plt.plot(epochs, np.log2(loss_values), label='Average Loss')
    plt.ylabel("Log2(Loss)", fontsize=12)
else:
    plt.plot(epochs, loss_values, label='Average Loss')

    plt.ylabel("Loss", fontsize=12)
plt.title(title, fontsize=14, )#fontweight='bold')
plt.xlabel("Epoch", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.savefig("62k_20_epochs.png")

#---------------------------------Combined plot-----------------
df_avg=pd.read_csv("results/10k_50_epochs/metrics/avg_loss.tsv",sep=' ',names=['run_id', 'avg_loss', 'epoch'])
df_log=pd.read_csv("results/10k_50_epochs/metrics/avg_log_loss.tsv",sep=' ',names=['run_id', 'avg_loss', 'epoch'])
df_rep=pd.read_csv("results/10k_50_epochs/metrics/avg_rep_loss.tsv",sep=' ',names=['run_id', 'avg_loss', 'epoch'])


log=True
loss_values = df_avg["avg_loss"]
loss_values_log = df_log["avg_loss"]
loss_values_rep = df_rep["avg_loss"]
print(loss_values)
epochs = range(1, len(loss_values) + 1)  # Epoch numbers
title="Loss for 10K batches with \n Logits and Representation Contribution per epoch"
plt.figure(figsize=(8, 6))
if log:
    plt.plot(epochs, np.log2(loss_values), label='Average Loss')
    plt.plot(epochs, np.log2(loss_values_log),  label='Logits Loss')
    plt.plot(epochs, np.log2(loss_values_rep),  label='Reps Loss')
    plt.ylabel("Log2(Loss)", fontsize=12)
else:
    plt.plot(epochs, loss_values, label='Average Loss')
    plt.plot(epochs, loss_values_log,  label='Logits Loss')
    plt.plot(epochs, loss_values_rep,  label='Reps Loss')
    plt.ylabel("Loss", fontsize=12)
plt.title(title, fontsize=14, )#fontweight='bold')
plt.xlabel("Epoch", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.savefig("log2_10k_50_epochs.png")
