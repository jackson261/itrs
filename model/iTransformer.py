import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
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
       # 添加时间维度的attention层 - 修改为与变量attention相同的结构
        self.time_encoder = Encoder(
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

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):                   #定义了 forecast 方法，用于执行时间序列预测。参数:x_enc：Encoder 的输入特征，形状为 [B, L, N]（批量大小、时间步数、特征数量）。
        if self.use_norm:                                                       #x_mark_enc：Encoder 的时间标记特征，例如时间戳信息，形状为 [B, L, F]（批量大小、时间步数、时间特征数量）。x_dec：Decoder 的输入特征。x_mark_dec：Decoder 的时间标记特征。
            # Normalization from Non-stationary Transformer       #归一化的目的：时间序列数据通常是非平稳的（即均值和方差随时间变化）。归一化可以缓解非平稳性对模型的影响，提高模型性能。
            means = x_enc.mean(1, keepdim=True).detach()          # 对 x_enc 的时间维度（dim=1）计算均值。keepdim=True 保留维度，确保结果形状为 [B, 1, N]。 detach() 使得均值在计算图之外，防止影响梯度。      
            x_enc = x_enc - means                                    #将输入数据 x_enc 的每个时间步减去均值，使其均值为 0。
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)   #计算 x_enc 在时间维度上的标准差。unbiased=False 表示不使用无偏估计。加入小值 1e-5 防止分母为零。
            x_enc /= stdev                                        #将 x_enc 除以标准差，使其方差为 1。

        _, _, N = x_enc.shape # B L N              #获取输入 x_enc 的形状：B：批量大小。L：时间步数。N：特征数量（变量数）。
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens                          #目的：将输入数据 x_enc 和时间标记特征 x_mark_enc 转换为高维表示（嵌入表示）。
                                                                                                                                            #形状变化：输入：[B, L, N]（批量大小、时间步数、特征数）。输出：[B, N, E]（批量大小、特征数、嵌入维度）。
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)                                                         #具体实现：嵌入层 self.enc_embedding 负责将时间步和特征嵌入到高维空间中。例如，可以通过线性变换、位置编码等方式实现。

        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules                  #目的：使用 Transformer 编码器对嵌入后的特征进行建模，提取时间序列中的时序关系。
        #enc_out, attns = self.encoder(enc_out, attn_mask=None)                                                                                    #输入：enc_out：嵌入后的输入特征 [B, N, E]。attn_mask=None：注意力掩码，用于屏蔽部分注意力（本例中未使用）。
        #处理变量间依赖
        var_out,attns=self.encoder(enc_out,attn_mask=None) #BNE ->BNE                                                                                                                                        #输出： enc_out：编码器的输出特征，形状仍为 [B, N, E]。attns：编码器中的注意力权重（可选，用于分析模型的注意力分布）。             
        # B N E -> B N S -> B S N 
        #处理时间依赖
       
        attn_mask = None  # 如果不需要使用掩码
       
        time_out=enc_out#直接使用enc_out,保持维度一致性
        time_out=self.time_encoder(time_out,attn_mask=attn_mask)[0]
        #time_out=self.time_norm(time_out)
   
        #特征融合（简单加权融合）
        fused_out=0.5*var_out+0.5*time_out #BNE
        # 根据预测长度动态调整融合权重
        #if self.pred_len == 336:
        #    weight = 0.5
        #elif self.pred_len < 336:
        #    weight = 0.4  # 短期预测更依赖变量特征
        #else:
        #    weight = 0.6  # 长期预测更依赖时间特征
        
        # 特征融合
        fused_out = weight * var_out + (1-weight) * time_out
        #使用融合后的特征进行预测
        dec_out=self.projector(fused_out).permute(0,2,1)[:,:,:N]
        
        #dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates                                            #使用投影层将编码器的输出特征映射到预测目标的维度。
                                                                                                                                     #self.projector 是一个线性层，将编码器输出的嵌入维度转换为预测步长。permute(0, 2, 1)交换维度，将形状 [B, N, S] 转换为 [B, S, N]，适应时间序列预测的格式。[:, :, :N]过滤掉非目标特征，仅保留目标特征的预测值。
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer                                            #恢复归一化后的预测结果到原始尺度。
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))                  #dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))将预测结果乘以标准差，恢复其方差。
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))                  # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)) 加回均值，恢复到原始尺度。          

        return dec_out                               #返回预测结果，形状为 [B, S, N]，其中：B：批量大小。S：预测时间步数。N：特征数量。

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):           #forward 是对 forecast 的简单包装。
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)             #作用：调用 forecast 方法生成预测结果。只返回最后 pred_len 时间步的预测值。
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]






#这里的forecast与exp_long_term_forecasting中predict有一个调用关系
#总结调用关系
#predict 调用 self.model：
#self.model() 会调用模型的 forward 方法。
#forward 调用 forecast：
#forecast 是核心预测逻辑，完成数据归一化、嵌入、编码、投影和反归一化。
#预测结果返回层层传递：
#forecast 的输出返回到 forward。
#forward 的输出返回到 predict。
#predict 处理结果：
#对预测结果进行后处理（如反归一化）并存储。
