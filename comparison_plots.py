import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logs_DQL_17 = pd.read_csv('model_logs/v17.1.csv')

fig, axs = plt.subplots(1, 1, figsize=(8, 8))

axs.plot(logs_DQL_17['iteration'][:200], logs_DQL_17['length_mean'][:200],
        label='Batch Size 64', color='skyblue')
axs.plot(logs_DQL_17['iteration'][:200], logs_DQL_17['length_mean'][:200],
        label='Batch Size 128', color='bisque')

axs.plot(logs_DQL_17['iteration'][9:200], logs_DQL_17['reward_mean'][9:200],
        label='Batch Size 64 Moving Average', color='blue')
axs.plot(logs_DQL_17['iteration'][9:200], logs_DQL_17['reward_mean'][9:200],
        label='Batch Size 128 Moving Average', color='red')

axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

plt.legend()
plt.show()