import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super(SpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.key_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        batch_size, num_step, N, _ = x.shape
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = attention / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention, dim=-1)

        out = torch.matmul(attention_weights, value)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.final_proj(out)

        return out


class TemporalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.key_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        batch_size, num_step, N, _ = x.shape
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)
        attention = attention / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention, dim=-1)

        out = torch.matmul(attention_weights, value)
        out = out.permute(0, 2, 1, 3)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.final_proj(out)

        return out


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(GatedFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_t = nn.Linear(hidden_dim, hidden_dim)
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, h_t, h_s):
        h_t = self.W_t(h_t)
        h_s = self.W_s(h_s)
        z = torch.sigmoid(self.W_g(torch.cat([h_t, h_s], dim=-1)))
        out = z * h_t + (1 - z) * h_s
        return out


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        support = self.linear(x)
        out = torch.matmul(adj, support)
        return out


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, lstm_out, gcn_out):
        attn_out, _ = self.multihead_attn(
            query=lstm_out,
            key=gcn_out,
            value=gcn_out,
            need_weights=False
        )

        fused_out = self.norm(lstm_out + self.alpha * attn_out)
        return fused_out


class AdaConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.3, support_len=2, order=2):
        super(GraphConv, self).__init__()
        self.nconv = NConv()
        self.order = order

        c_in_total = (order * support_len + 1) * c_in

        self.mlp = nn.Conv2d(c_in_total, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class STGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes=< YOUR_NODE_NUM >,
                 num_heads=8, lstm_layers=2, dropout_rate=0.1,
                 gcn_order=2, use_adaptive_adj=True, aptinit=None):
        super(STGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.num_nodes = num_nodes
        self.use_adaptive_adj = use_adaptive_adj

        self.spatial_attention = SpatialAttention(input_dim, hidden_dim, num_heads)
        self.temporal_attention = TemporalAttention(input_dim, hidden_dim, num_heads)

        if use_adaptive_adj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)

        self.static_gcn = GraphConv(hidden_dim, hidden_dim, dropout=dropout_rate,
                                    support_len=1, order=gcn_order)

        if use_adaptive_adj:
            self.adaptive_gcn = GraphConv(hidden_dim, hidden_dim, dropout=dropout_rate,
                                          support_len=1, order=gcn_order)

        self.gcn_gate = GatedFusion(hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        self.cross_attn = CrossAttentionFusion(hidden_dim, num_heads)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

        self.residual_proj = None
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj, return_adj=False):
        batch_size, seq_len, nodes, _ = x.shape
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        spatial_out = self.spatial_attention(x)
        spatial_out_gcn = spatial_out.permute(0, 3, 2, 1)

        supports = [adj]
        adaptive_adj = None
        if self.use_adaptive_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            supports.append(adp)
            adaptive_adj = adp

        static_out = self.static_gcn(spatial_out_gcn, supports[:1])
        static_out = static_out.permute(0, 3, 2, 1)

        if self.use_adaptive_adj:
            adaptive_out = self.adaptive_gcn(spatial_out_gcn, supports[1:])
            adaptive_out = adaptive_out.permute(0, 3, 2, 1)
            gcn_out = self.gcn_gate(static_out, adaptive_out)
        else:
            gcn_out = static_out

        temporal_out = self.temporal_attention(x)
        temporal_out = self.layer_norm(temporal_out + residual)

        lstm_in = temporal_out.permute(0, 2, 1, 3)
        lstm_in = lstm_in.reshape(batch_size * nodes, seq_len, -1)

        h0 = torch.zeros(self.lstm_layers, lstm_in.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.lstm_layers, lstm_in.size(0), self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(lstm_in, (h0, c0))

        gcn_reshaped = gcn_out.permute(0, 2, 1, 3)
        gcn_reshaped = gcn_reshaped.reshape(batch_size * nodes, seq_len, -1)

        fused_features = self.cross_attn(lstm_out, gcn_reshaped)
        last_hidden = fused_features[:, -1, :]
        last_hidden = last_hidden.reshape(batch_size, nodes, -1)

        output = self.output_layer(last_hidden)

        if return_adj:
            return output, adaptive_adj
        return output