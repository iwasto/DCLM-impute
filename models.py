import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads=4,
                 attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):
        super(SelfAttention, self).__init__()
        self.attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.AE = nn.Linear(input_size, hidden_size)

        self.query_head = nn.ModuleList()
        self.key_head = nn.ModuleList()
        self.value_head = nn.ModuleList()
        for i in range(num_attention_heads):
            self.query_head.append(nn.Linear(hidden_size, hidden_size))
            self.key_head.append(nn.Linear(hidden_size, hidden_size))
            self.value_head.append(nn.Linear(hidden_size, hidden_size))

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense1 = nn.Linear(hidden_size, hidden_size*4)

        self.ZINB_Encoder = nn.Sequential(nn.Linear(hidden_size*4, 512), nn.ReLU(),nn.Linear(512, 256), nn.ReLU())        #做一个Autoencoder based on ZINB distribution ！！！
        self.pi_Encoder =  nn.Sequential(nn.Linear(256, input_size),nn.Sigmoid())
        self.disp_Encoder = nn.Sequential(nn.Linear(256, input_size), nn.Softplus())
        self.mean_Encoder = nn.Linear(256, input_size)
  
        self.dense2 = nn.Linear(hidden_size*4,hidden_size*2)
        self.dense3 = nn.Linear(hidden_size*2,hidden_size)
        self.dense4 = nn.Linear(hidden_size,hidden_size)
         
        self.classifier = nn.Sequential(  #新增的分类器！！
            nn.Linear(hidden_size, 7),
            nn.LogSoftmax(dim=1)
        )

        
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        self.decoder = nn.Sequential(            #生成器！与原数据做对比
            nn.Linear(hidden_size,hidden_size*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size*2,input_size),
            nn.ReLU(inplace=True) 
        )

    def clip_by_tensor(self,t, t_min, t_max):  #对输入张量 t 进行裁剪（clipping），使其值限制在 [t_min, t_max] 的范围内。如果 t 中的某个元素小于 t_min，则将其替换为 t_min；如果某个元素大于t_max，则将其替换为t_max
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        t = torch.tensor(t,dtype = torch.float32)
        t_min = torch.tensor(t_min,dtype = torch.float32)
        t_max = torch.tensor(t_max,dtype = torch.float32)

        result = torch.tensor((t >= t_min),dtype = torch.float32) * t + torch.tensor((t < t_min),dtype = torch.float32) * t_min
        result = torch.tensor((result <= t_max),dtype = torch.float32) * result + torch.tensor((result > t_max),dtype = torch.float32) * t_max
        return result
    
    def forward(self, input_tensor, return_attention=False, is_drop=False):

        # [cell, emb] <- [cell, genes]
        input_tensor = self.AE(input_tensor)
        #
        outputs = []
        for i in range(self.attention_heads):
            query = self.query_head[i]
            key = self.key_head[i]
            value = self.value_head[i]

            query_layer = query(input_tensor)
            key_layer = key(input_tensor)
            value_layer = value(input_tensor)

            # [cells, cells] = [cell, emb]*[emb, cell]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(
                -1, -2))

            attention_scores = attention_scores / math.sqrt(
                self.hidden_size)

            if return_attention:
                return attention_scores

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            if is_drop:
                attention_probs = self.attn_dropout(attention_probs)
            # [cells, emb] = [cells, cells] * [cells, emb]
            context_layer = torch.matmul(attention_probs, value_layer)

            outputs.append(context_layer)
        # avg([heads, cells, emb])
        output = torch.mean(torch.stack(outputs), 0)

        hidden_states = self.dense1(output)
         
        z = self.ZINB_Encoder(hidden_states)     # Autoencoder based on ZINB distribution!!!
        pi = self.pi_Encoder(z)
        disp = self.disp_Encoder(z)
        disp = self.clip_by_tensor(disp,1e-4,1e4)
        mean = self.mean_Encoder(z)
        mean = self.clip_by_tensor(torch.exp(mean),1e-5,1e6)
        
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dense3(hidden_states)

        dc_out = self.classifier(hidden_states)  #分类器

        hidden_states = self.dense4(hidden_states)
        
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        dc_decode = self.decoder(hidden_states)
        return hidden_states, dc_out ,dc_decode,pi,disp,mean

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)
        self.to(self.args.device)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class MultiScaleConv(nn.Module):
    def __init__(self, hidden_size):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(1, hidden_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, hidden_size, kernel_size=5, padding=2)
        self.fusion = nn.Linear(3 * hidden_size, hidden_size)

    def forward(self, x):
        # x shape: [cell, hidden_size]
        x = x.unsqueeze(1)  # [cell, 1, hidden_size]
        
        # 多尺度卷积
        scale1 = F.relu(self.conv1(x))  # [cell, hidden_size, hidden_size]
        scale3 = F.relu(self.conv3(x))  # [cell, hidden_size, hidden_size]
        scale5 = F.relu(self.conv5(x))  # [cell, hidden_size, hidden_size]
        
        # 拼接多尺度特征
        multi_scale = torch.cat([scale1, scale3, scale5], dim=1)  # [cell, 3*hidden_size, hidden_size]
        multi_scale = multi_scale.permute(0, 2, 1)  # [cell, hidden_size, 3*hidden_size]
        
        # 特征融合
        fused = self.fusion(multi_scale)  # [cell, hidden_size, hidden_size]
        return fused.mean(dim=1)  # 全局平均 -> [cell, hidden_size]

class AttentionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads=4,
                 attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):
        super(AttentionEncoder, self).__init__()

        self.attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.AE = nn.Linear(input_size, hidden_size)
        self.multi_scale_conv = MultiScaleConv(hidden_size)  # 多尺度卷积模块

        self.query_head = nn.ModuleList()
        self.key_head = nn.ModuleList()
        self.value_head = nn.ModuleList()
        for i in range(num_attention_heads):
            self.query_head.append(nn.Linear(hidden_size, hidden_size))
            self.key_head.append(nn.Linear(hidden_size, hidden_size))
            self.value_head.append(nn.Linear(hidden_size, hidden_size))

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        # 前馈网络
        self.dense1 = nn.Linear(hidden_size, hidden_size*4)
        self.dense2 = nn.Linear(hidden_size*4, hidden_size*2)
        self.dense3 = nn.Linear(hidden_size*2, hidden_size)
        self.dense4 = nn.Linear(hidden_size, hidden_size)
         
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor, return_attention=False):
        # [cell, emb] <- [cell, genes]
        input_tensor = self.AE(input_tensor)
        input_tensor = self.multi_scale_conv(input_tensor)  # 应用多尺度特征提取
        
        if return_attention:
            all_attentions = []
            for i in range(self.attention_heads):
                query = self.query_head[i]
                key = self.key_head[i]
                value = self.value_head[i]

                query_layer = query(input_tensor)
                key_layer = key(input_tensor)
                value_layer = value(input_tensor)

                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(self.hidden_size)
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
                if self.training:
                    attention_probs = self.attn_dropout(attention_probs)
                all_attentions.append(attention_probs)
            return torch.mean(torch.stack(all_attentions), 0)

        outputs = []
        for i in range(self.attention_heads):
            query = self.query_head[i]
            key = self.key_head[i]
            value = self.value_head[i]

            query_layer = query(input_tensor)
            key_layer = key(input_tensor)
            value_layer = value(input_tensor)

            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.hidden_size)
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            if self.training:
                attention_probs = self.attn_dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)
            outputs.append(context_layer)
            
        # 平均多头输出
        output = torch.mean(torch.stack(outputs), 0)

        # 前馈网络
        hidden_states = self.dense1(output)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dense3(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dense4(hidden_states)
        
        # Dropout和残差连接
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

# -------------------------- BYOLModel --------------------------
class BYOLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, proj_dim=128):
        super().__init__()
        # Online网络
        self.online_encoder = AttentionEncoder(input_dim, hidden_dim)
        self.online_projector = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )
        
        # Target网络
        self.target_encoder = AttentionEncoder(input_dim, hidden_dim)
        self.target_projector = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim)
        )
        
        # 初始化目标网络
        self._init_target()
    
    def _init_target(self):
        for o, t in zip(self.online_encoder.parameters(), 
                       self.target_encoder.parameters()):
            t.data.copy_(o.data)
            t.requires_grad = False
        for o, t in zip(self.online_projector.parameters(),
                       self.target_projector.parameters()):
            t.data.copy_(o.data)
            t.requires_grad = False
    
    @torch.no_grad()
    def update_target(self, decay=0.996):
        for o, t in zip(self.online_encoder.parameters(),
                      self.target_encoder.parameters()):
            t.data = decay*t.data + (1-decay)*o.data
        for o, t in zip(self.online_projector.parameters(),
                      self.target_projector.parameters()):
            t.data = decay*t.data + (1-decay)*o.data
    
    def forward(self, x1, x2):
        # Online网络
        online_z = self.online_encoder(x1)
        online_proj = self.online_projector(online_z)
        online_pred = self.predictor(online_proj)
        
        # Target网络
        with torch.no_grad():
            target_z = self.target_encoder(x2)
            target_proj = self.target_projector(target_z)
        
        return online_pred, target_proj.detach(), online_z
        
    def get_representation(self, x):
        """
        获取输入数据的嵌入表示（在线编码器的输出）
        
        参数:
            x: 输入张量 [batch_size, input_dim]
        
        返回:
            representation: 嵌入表示 [batch_size, hidden_dim]
        """
        with torch.no_grad():
            representation = self.online_encoder(x)
        return representation