import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logs_DQL_17 = pd.read_csv('model_logs/v17.1.csv')



axs.plot(logs_DQL_17['iteration'][:200], logs_DQL_17['length_mean'][:200],
        label='Batch Size 64', color='skyblue')
axs.plot(logs_DQL_17['iteration'][:200], logs_DQL_17['length_mean'][:200],
        label='Batch Size 128', color='bisque')

axs.plot(logs_DQL_17['iteration'][9:200], logs_DQL_17['length_mean_ma'][9:200],
        label='Batch Size 64 Moving Average', color='blue')
axs.plot(logs_DQL_17['iteration'][9:200], logs_DQL_17['length_mean_ma'][9:200],
        label='Batch Size 128 Moving Average', color='red')

axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

plt.legend()
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.set_title('Snake Mean Length vs PreTraining')

axs.plot(df_base['iteration'][:100], df_base['length_mean'][:100], 
        label='DQN', color='skyblue')
axs.plot(df_super['iteration'][:100], df_super['length_mean'][:100], 
        label='DQN PreTrained', color='bisque')

axs.plot(df_base['iteration'][9:100], df_base['length_mean_ma'][9:100], 
        label='DQN Moving Average', color='blue')
axs.plot(df_super['iteration'][9:100], df_super['length_mean_ma'][9:100], 
        label='DQN PreTrained Moving Average', color='red')

axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

plt.legend()
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.set_title('Snake Mean Length vs Reward Type')

axs.plot(df_base['iteration'][:100], df_base['length_mean'][:100], 
        label='Static Reward', color='skyblue')
axs.plot(df_reward['iteration'][:100], df_reward['length_mean'][:100], 
        label='Length Dependent Reward', color='bisque')

axs.plot(df_base['iteration'][9:100], df_base['length_mean_ma'][9:100], 
        label='Static Reward Moving Average', color='blue')
axs.plot(df_reward['iteration'][9:100], df_reward['length_mean_ma'][9:100], 
        label='Length Dependent Reward Moving Average', color='red')

axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

plt.legend()
plt.show()


fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.set_title('Snake Mean Length vs Reward Type')

axs.plot(df_base['iteration'][:100], df_base['loss'][:100], 
        label='Static Reward', color='skyblue')
axs.plot(df_reward['iteration'][:100], df_reward['loss'][:100], 
        label='Length Dependent Reward', color='bisque')

axs.plot(df_base['iteration'][9:100], df_base['loss_ma'][9:100], 
        label='Static Reward Moving Average', color='blue')
axs.plot(df_reward['iteration'][9:100], df_reward['loss_ma'][9:100], 
        label='Length Dependent Reward Moving Average', color='red')

axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

plt.legend()
plt.show()
