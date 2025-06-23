from math import sqrt

import torch
import torch.nn as nn
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output
    
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars
    
class TS_Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):

        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.num_tokens = 1000
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        super(TS_Model, self).__init__()

    def forward(self, x_enc):
        enc_out = self.forecast(x_enc)
        return enc_out

    def forecast(self, x_enc):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
    
        return enc_out
    
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding