import torch
import numpy as np

def select_neighbours(hidden, k):
    with torch.no_grad():
        sim = torch.cosine_similarity(hidden.unsqueeze(1), hidden.unsqueeze(0), dim=-1)
        sim = sim.fill_diagonal_(0.0)
        choose_cell = sim.argsort()[:, -k:].to('cpu').numpy()
    return choose_cell

def LS_imputation(drop_data, choose_cell, device, filter_noise=2):
    original_data = torch.FloatTensor(np.copy(drop_data)).to(device)
    dataImp = original_data.clone().to(device)

    for i in range(dataImp.shape[0]):
        nonzero_index = dataImp[i].nonzero()
        zero_index = (dataImp[i] == 0).nonzero()
        y = original_data[i, nonzero_index]
        x = original_data[choose_cell[i], nonzero_index]

        xtx = torch.matmul(x.T, x)
        rank = torch.linalg.matrix_rank(xtx)
        if rank != x.shape[-1]: # detect the singular matrix
            print('It is a singular matrix, use average imputation')
            return Average_imputation(drop_data, choose_cell, device, filter_noise)

        w = torch.matmul(torch.matmul(torch.linalg.inv(xtx), x.T), y)
        impute_data = torch.matmul(original_data[choose_cell[i], zero_index], w)
        impute_data[impute_data <= filter_noise] = 0   # filter noise
        dataImp[i, zero_index] = impute_data

    return dataImp.detach().cpu().numpy()

def Average_imputation(drop_data, choose_cell, device, filter_noise=2):
    original_data = torch.FloatTensor(np.copy(drop_data)).to(device)
    dataImp = original_data.clone().to(device)
    for i in range(dataImp.shape[0]):
        zero_index = (dataImp[i] == 0).nonzero()

        impute_data = torch.mean(original_data[choose_cell[i], zero_index], dim=1)
        # filter noise
        impute_data[impute_data <= filter_noise] = 0
        dataImp[i, zero_index] = impute_data.unsqueeze(-1)
    return dataImp.detach().cpu().numpy()