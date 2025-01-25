import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted,  DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # print("ratio" , configs.d_model/ configs.n_heads)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_model * 2,
            dropout=0.1,
            activation="relu"
        )
        
        # Stack several layers in nn.TransformerDecoder
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=configs.e_layers)
        
        
        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projector = nn.Sequential(
            
        nn.Linear(configs.d_model, configs.d_model*2),
        nn.ReLU(),
       
        
        nn.Linear( configs.d_model*2, configs.c_out)
)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # print(N,"Nnnnnnnnn")
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_in = self.dec_embedding(x_dec, x_mark_dec)

        # print("tgt shape:", dec_in.shape)
        # print("mem shape", enc_out.shape)
        tgt=dec_in
        memory=enc_out

        # print("Before transpose:")
        # print("tgt shape:", tgt.shape)
        # print("memory shape:", memory.shape)

        # Apply transpose
        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)

        # print("After transpose:")
        # print("tgt shape:", tgt.shape)
        # print("memory shape:", memory.shape)


        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None,
            memory_mask=None
        )

        dec_out = self.projector(dec_out)  # [B, 72, 172]
        dec_out = dec_out[:, :, :N] 
        # # print("dec_out shape:", dec_out.shape)  # Expected: [batch_size, pred_len, feature_dim]
        # # print("stdev shape:", stdev.shape)      # Expected: [batch_size, 1, feature_dim]
        # # print("repeated stdev shape:", stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1).shape)
        # if self.use_norm:
        # # Ensure stdev matches dec_out dimensions
        #     stdev = stdev.repeat(1, dec_out.shape[0], 1)  # [32, pred_len, 512]
        #     dec_out = dec_out * stdev
        #     dec_out = dec_out + means.repeat(1, dec_out.shape[0], 1)  # [32, pred_len, 512]



        # print("here hrere 1 ", dec_out.shape)
        
        # dec_out = self.projector(dec_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        # print("here hrere ", dec_out.shape)

        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #     dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        if self.use_norm:   
            # print("dec_out shape", dec_out.shape)
            # print("stdev shape", stdev.shape) 
            # print("means shape", means.shape) 

            dec_out = dec_out.permute(1, 0, 2) 
            # dec_out = dec_out * stdev + means
            stdev_72 = stdev#[:, :, :72]   # [32, 1, 72]
            means_72 = means#[:, :, :72]   # [32, 1, 72]

            stdev_72 = stdev_72.repeat(1, dec_out.shape[1], 1)  

            # print("dec_out shape", dec_out.shape)
            # print("stdev shape", stdev_72.shape) 
            # print("means shape", means_72.shape) 

            # Then dec_out * stdev_72 + means_72 will work
            dec_out = dec_out * stdev_72 + means_72
            


        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]