import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

def calculate_accuracy(y_pred, y):
  _, predicted = torch.max(y_pred, 1)
  acc = (predicted == y).sum() / len(y)
  return acc

def visualize(history_path, viz_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    acc_path = os.path.join(viz_path, 'accuracy.png')
    loss_path = os.path.join(viz_path, 'loss.png')
    
    X_axis = range(0, epochs, 2)

    if not os.path.exists(viz_path):
        print('make directory')
        os.makedirs(viz_path)
        
    history = pd.read_csv(history_path)

    ax1.plot(history['accuracy'], 'b', label='Training acc')
    ax1.set_xticks(X_axis)
    ax1.legend(loc="center right")
    ax1.set_title('Training accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')

    ax2.plot(history['loss'], 'b', label='Training loss')
    ax2.legend(loc="center right")
    ax2.legend(loc="center right")
    ax2.set_title('Training loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')

    extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(acc_path, bbox_inches=extent1.expanded(1.3, 1.25))
    fig.savefig(loss_path, bbox_inches=extent2.expanded(1.3, 1.25))
    fig.show()