import torch
import numpy as np

def data_augmentations(X, rate=0.4):
    X_aug = X.clone().to(X.device)
    select_index = X_aug.nonzero()
    
    if len(select_index) > 0:
        ix = np.random.choice(
            range(len(select_index)), 
            int(np.floor(rate * len(select_index))), 
            replace=False
        )
        X_aug[select_index.T[0][ix], select_index.T[1][ix]] = 0.0
    
    return X_aug