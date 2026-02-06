
import torch
import numpy as np
import pandas as pd
from preprocessing import load_data, impute_dropout, get_cluster_label, take_norm
from models import SelfAttention, BYOLModel, AttentionEncoder
from training import training_simclr, train_byol
from imputation import select_neighbours, LS_imputation
import warnings

warnings.filterwarnings("ignore")

# 设备设置
device = torch.device('cpu')

# 数据加载和处理
dataset_name = 'Zeisel'
drop_rate = 0.1
groundTruth_data, cells, genes = load_data('Zeisel')
drop_data = impute_dropout(groundTruth_data, drop_rate=drop_rate)
print(f'dataset: {dataset_name}, drop rate: {drop_rate}')

# 保存dropout数据
pd.DataFrame(drop_data.T, index=genes, columns=cells).to_csv('0.1_dropout_dentategyrus.csv')

# 数据标准化
cell_row_data = take_norm(drop_data)
tensor_norm_data = torch.tensor(cell_row_data, dtype=torch.float32)

# 获取聚类标签
num_cell = cell_row_data.shape[0]
num_gene = cell_row_data.shape[1]
_, pre_label = get_cluster_label(
    cell_row_data, 
    n_cluster=7, 
    n_cell=num_cell, 
    n_gene=num_gene
)
pre_label_tensor = torch.from_numpy(pre_label).long()

# 训练模型
X = torch.FloatTensor(np.copy(drop_data)).to(device)

# 训练BYOL模型
model2 = train_byol(X, hidden_size=256, epochs=100, aug_rate=0.4)
hidden_states_2 = model2.get_representation(X)

# 训练SimCLR模型
model1 = training_simclr(
    X, 
    pre_label_tensor, 
    hidden_size=256, 
    epochs=100, 
    aug_rate=0.4
)
with torch.no_grad():
    hidden_states, dc_out, dc_decode, _, _, _ = model1(X)

# 组合特征
hidden_states_new = hidden_states + 0.25 * hidden_states_2

# 选择邻居并进行插补
choose_cell = select_neighbours(hidden_states_new, k=20)
imputed_data = LS_imputation(drop_data, choose_cell, device)

# 保存结
pd.DataFrame(imputed_data.T, index=genes, columns=cells).to_csv('impute_0.4.csv')