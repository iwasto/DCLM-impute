import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import SelfAttention, BYOLModel
from losses import ConstrastiveLoss, ZINB, _cosine_loss
from augmentations import data_augmentations

def training_simclr(X, label, hidden_size=128, epochs=100, aug_rate=0.4):  # 修复参数名: epoch -> epochs
    model = SelfAttention(input_size=X.shape[-1], hidden_size=hidden_size).to(X.device)
    criterion_instance = ConstrastiveLoss(X.shape[0], 1.5)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion_cls = torch.nn.NLLLoss()  # 更清晰的命名
    criterion_recon = torch.nn.MSELoss()  # 更清晰的命名
    
    for epoch in range(epochs):  # 使用epoch而不是i
        model.train()
        optimizer.zero_grad()

        # 应用数据增强
        X_aug1 = data_augmentations(X, rate=aug_rate)
        X_aug2 = data_augmentations(X, rate=aug_rate)
        
        # 前向传播
        y1, dc_1, decode_1, pi_1, disp_1, mean_1 = model(X_aug1)  
        y2, dc_2, decode_2, pi_2, disp_2, mean_2 = model(X_aug2) 

        
        zinb = ZINB(pi_1, theta=disp_1, ridge_lambda=0, debug=False)
        zinb_loss = zinb.loss(X, mean_2, mean=True)
        
        # 计算总损失
        loss = (
            0.4 * criterion_instance(y1, y2) +
            0.3 * criterion_cls(dc_1, label) +
            0.2 * zinb_loss +
            0.1 * criterion_recon(decode_1, decode_2)
        )
        
        # 打印训练进度 - 使用正确的变量名
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
        # 反向传播
        loss.backward()
        optimizer.step()
        
    print("SimCLR training completed")
    return model

def train_byol(X, hidden_size=128, epochs=100, aug_rate=0.1):
    device = X.device
    model = BYOLModel(input_dim=X.shape[1], hidden_dim=hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 应用数据增强
        x1 = data_augmentations(X, rate=aug_rate)
        x2 = data_augmentations(X, rate=aug_rate)
        
        # 前向传播
        online_pred1, target_proj2, _ = model(x1, x2)
        online_pred2, target_proj1, _ = model(x2, x1)
        
        # 计算对称损失
        loss = 0.5 * (
            _cosine_loss(online_pred1, target_proj2) +
            _cosine_loss(online_pred2, target_proj1)
        )
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target()
        scheduler.step()
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    print("Byol training completed")
    
    return model