import torch
import torch.nn as nn
from config import R_SCALE

def decode_from_direct_space(x_y_r_norm, area_targets, graph_scale):
    # x_y_r_norm[..., :2] is already in (0, 1) thanks to sigmoid
    x = x_y_r_norm[..., 0] * graph_scale
    y = x_y_r_norm[..., 1] * graph_scale

    # r_norm is in (-1, 1) thanks to tanh
    r = x_y_r_norm[..., 2] * R_SCALE

    w = torch.sqrt(area_targets) * torch.exp(r / 2.0)
    h = torch.sqrt(area_targets) * torch.exp(-r / 2.0)
    return torch.stack([x, y, w, h], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, src_key_padding_mask, attn_bias):
        # Classic Pre-Norm architecture
        x_norm = self.norm1(x)

        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_bias,
            need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FloorplanDirectTransformer(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=6, num_heads=8):
        super().__init__()
        self.num_heads = num_heads

        # Input size 14:
        # 1 (Area) + 5 (Constraints) + 3 (Spectral) + 2 (Pin CoM) + 1 (Known R) + 2 (Known X, Y)
        self.feature_proj = nn.Linear(14, hidden_dim)

        # 21 classes: distances from 0 to 20
        self.spatial_embedding = nn.Embedding(21, num_heads)

        self.pin_spatial_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )

        self.edge_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, num_heads)
        )
        self.edge_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        # Predict 3 parameters: [x_norm, y_norm, r_norm]
        self.head = nn.Linear(hidden_dim, 3)

        # Initialize with zeros, so model outputs center initially (if sigmoid is used) or zeros
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, node_features, padding_mask=None, b2b_matrix=None, p2b_matrix=None, pins_pos=None, spd_matrix=None):
        x = self.feature_proj(node_features)

        if p2b_matrix is not None and pins_pos is not None:
            pin_embeddings = self.pin_spatial_proj(pins_pos)
            pin_anchors = torch.bmm(p2b_matrix, pin_embeddings)
            x = x + pin_anchors

        attn_bias = None
        if spd_matrix is not None:
            B, K, _ = spd_matrix.shape

            # 1. Get topological Spatial Bias from shortest paths matrix
            # shape: (B, K, K, num_heads) -> (B, num_heads, K, K)
            spatial_bias = self.spatial_embedding(spd_matrix)
            spatial_bias = spatial_bias.permute(0, 3, 1, 2)

            # 2. Add info about direct connections strength (from b2b_matrix)
            if b2b_matrix is not None:
                eye = torch.eye(K, device=b2b_matrix.device).unsqueeze(0)
                b2b_matrix_adj = b2b_matrix + eye
                b2b_matrix_adj = torch.log1p(b2b_matrix_adj)
                b2b_features = b2b_matrix_adj.unsqueeze(-1)

                edge_bias = self.edge_proj(b2b_features)
                edge_bias = edge_bias.permute(0, 3, 1, 2)
                edge_bias = edge_bias * self.edge_scale

                # Sum topology (SPD) and connection strength
                spatial_bias = spatial_bias + edge_bias

            # Reshape to meet MultiheadAttention requirements: (B * num_heads, K, K)
            attn_bias = spatial_bias.reshape(B * self.num_heads, K, K)

        for block in self.blocks:
            x = block(x, padding_mask, attn_bias)

        x = self.final_norm(x)
        out = self.head(x)

        # Separate coordinates and r
        xy_norm = torch.sigmoid(out[..., :2])  # x, y in range (0, 1)
        r_norm = torch.tanh(out[..., 2:3])  # r_norm in range (-1, 1)

        # =====================================================================
        # ARCHITECTURAL FIXATION OF MIB GROUPS (Training)
        # =====================================================================
        is_fixed = node_features[..., 1] == 1.0
        is_preplaced = node_features[..., 2] == 1.0
        is_hard = is_fixed | is_preplaced
        mib_ids = node_features[..., 3]  # Index 3 in node_features is the MIB group ID

        B, N, _ = r_norm.shape
        updated_r_norm = r_norm.clone()

        for b in range(B):
            curr_mib = mib_ids[b]
            curr_hard = is_hard[b]
            unique_ids = curr_mib[curr_mib > 0].unique()

            for gid in unique_ids:
                mask = (curr_mib == gid)
                if mask.sum() > 1:
                    hard_in_group = mask & curr_hard
                    if hard_in_group.any():
                        # SCENARIO A: There is a hard block. Take r_norm from the first hard element.
                        ref_idx = torch.where(hard_in_group)[0][0]
                        updated_r_norm[b, mask] = r_norm[b, ref_idx]
                    else:
                        # SCENARIO B: No hard blocks. Average the entire group.
                        updated_r_norm[b, mask] = r_norm[b, mask].mean()

        r_norm = updated_r_norm
        # =====================================================================

        return torch.cat([xy_norm, r_norm], dim=-1)

