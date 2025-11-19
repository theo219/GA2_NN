import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logs_pytorch = pd.read_csv('model_logs/v17.1.csv')
logs_tf = pd.read_csv('model_logs/v17.1_tf.csv')

colors = {
    "length_mean" : ("bisque","red"),
    "reward_mean" : ("skyblue","blue"),
    "loss" : ("lightgreen","green"),
}

for idx, category in zip(['length_mean','reward_mean','loss'],['Mean Length', 'Mean Reward', 'Loss']):
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))

    axs.plot(logs_tf['iteration'], logs_tf[idx],
         label='Tensorflow', color=colors[idx][0])

    axs.plot(logs_pytorch['iteration'], logs_pytorch[idx],
         label='Pytorch', color=colors[idx][1])

    axs.set(xlabel='Iteration', ylabel=category)
    plt.legend()
    plt.show()