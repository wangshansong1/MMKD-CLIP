import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiCLIP_AE(nn.Module):
    def __init__(self, clip_dims, latent_dim, h_dim, z_dim, dropout_rate=0.2, noise_std=0.01):
        super().__init__()

        self.latent_dim = latent_dim
        self.z_dim, self.dropout_rate, self.noise_std = z_dim, dropout_rate, noise_std

        self.proj_layers = nn.ModuleDict({
            name: nn.Linear(in_dim, latent_dim) for name, in_dim in clip_dims.items()
        })

        self.decoder_proj_layers = nn.ModuleDict({
            name: nn.Linear(latent_dim, in_dim) for name, in_dim in clip_dims.items()
        })

        self.visual_encoder = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )


        self.visual_decoder = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, latent_dim),
        )

        self.text_decoder = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, latent_dim),
        )
    def add_noise(self, x):
        if self.training and self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x
    
    def visual_encode(self, x):
        x = self.add_noise(x)
        return self.visual_encoder(x)
    
    def text_encode(self, x):
        x = self.add_noise(x)
        return self.text_encoder(x)
    
    def visual_decode(self, z):
        return self.visual_decoder(z)
    
    def text_decode(self, z):
        return self.text_decoder(z)

    def forward(self, image_feats, text_feats, clip_names_batch):

        device = next(self.parameters()).device
        
        # ============ 1) Flatten ============
        # 把 (sample_idx, img_feat, txt_feat, clip_name) 展平到一维列表
        all_img_feats   = []
        all_txt_feats   = []
        all_clip_names  = []
        all_sample_idx  = []   # 记录属于哪个样本

        for s_idx, (img_list, txt_list, c_names) in enumerate(zip(image_feats, text_feats, clip_names_batch)):
            for img_f, txt_f, cn in zip(img_list, txt_list, c_names):
                all_img_feats.append(img_f)
                all_txt_feats.append(txt_f)
                all_clip_names.append(cn)
                all_sample_idx.append(s_idx)

        M = len(all_img_feats)  # flatten 后共有 M 条 clip
        if M == 0:
            # 空batch，返回全0
            zero_t = torch.tensor(0.0, device=device)
            return zero_t, zero_t, zero_t

        # 张量化 sample_idx，后面对比损失要用
        all_sample_idx = torch.tensor(all_sample_idx, device=device, dtype=torch.long)

        # ============ 缓存表，用于保存并行处理后的结果 ============
        # 存重建向量 (用来算recon loss)
        x_recon_img_buf = [None]*M
        x_recon_txt_buf = [None]*M

        # ============ 2) 按 clip_name 分组并并行处理 ============
        from collections import defaultdict
        clip_groups = defaultdict(list)
        for i, cn in enumerate(all_clip_names):
            clip_groups[cn].append(i)

        for cn, indices in clip_groups.items():
            # 收集同 clip_name 的所有图文特征, shape [N_c, feat_dim]
            img_batch_list = [all_img_feats[idx].to(device) for idx in indices]
            txt_batch_list = [all_txt_feats[idx].to(device) for idx in indices]

            # 堆叠成大张量
            img_batch_t = torch.stack(img_batch_list, dim=0)  # [N_c, in_dim]
            txt_batch_t = torch.stack(txt_batch_list, dim=0)  # [N_c, in_dim]

            # ---------- 编码 -----------
            x_proj_img = self.proj_layers[cn](img_batch_t)
            z_img = self.visual_encode(x_proj_img)

            x_proj_txt = self.proj_layers[cn](txt_batch_t)
            z_txt = self.text_encode(x_proj_txt)

            # ---------- 解码 & 重建 -----------
            x_dec_img = self.visual_decode(z_img)
            x_recon_img_ = self.decoder_proj_layers[cn](x_dec_img)

            x_dec_txt = self.text_decode(z_txt)
            x_recon_txt_ = self.decoder_proj_layers[cn](x_dec_txt)

            # ---------- 存入 buffer -----------
            for local_i, global_i in enumerate(indices):
                x_recon_img_buf[global_i] = x_recon_img_[local_i]
                x_recon_txt_buf[global_i] = x_recon_txt_[local_i]


        recon_losses = []
        for i in range(M):
            # 与原始 all_img_feats[i], all_txt_feats[i] 做 MSE
            recon_img = F.mse_loss(x_recon_img_buf[i], all_img_feats[i].to(device))
            recon_txt = F.mse_loss(x_recon_txt_buf[i], all_txt_feats[i].to(device))
            recon_losses.append(recon_img)
            recon_losses.append(recon_txt)
        if len(recon_losses) > 0:
            loss_recon = torch.stack(recon_losses).mean()
        else:
            loss_recon = torch.tensor(0.0, device=device)

        return loss_recon
    
    def visual_forward(self, feat_img, clip_name):
        x_proj_img = self.proj_layers[clip_name](feat_img)
        # z_img = self.visual_encode(x_proj_img)
        # z_p_img = self.visual_decode(z_img)
        return x_proj_img
    
    def text_forward(self, feat_txt, clip_name):
        x_proj_txt = self.proj_layers[clip_name](feat_txt)
        # z_txt = self.text_encode(x_proj_txt)
        # z_p_txt = self.text_decode(z_txt)
        return x_proj_txt



    
def nt_xent_loss(z, sample_idx, temperature=0.07):
    """
    z: shape [N, dim], all embeddings stacked
    sample_idx: shape [N], which sample each embedding belongs to
    temperature: float
    InfoNCE/NT-Xent: same sample_idx => positives; different => negative
    """
    z_norm = F.normalize(z, p=2, dim=1)   # [N, dim]
    sim_matrix = torch.matmul(z_norm, z_norm.t())  # [N, N]

    sim_matrix = sim_matrix / temperature

    
    mask = (sample_idx.unsqueeze(1) == sample_idx.unsqueeze(0)).float().to(z.device)
    
    mask_self = torch.eye(len(z), device=z.device)
    mask = mask - mask_self

    log_sum_exp = F.log_softmax(sim_matrix, dim=1)
    
    positives_logprob = (log_sum_exp * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    loss = - positives_logprob.mean()
    return loss