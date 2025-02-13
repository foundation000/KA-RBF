import torch
from torch import nn
import torch.nn.functional as F


import re
import time
import pickle
import torch.optim as optim

# -------------------------------------------------- #
#（1）muti_head_selfattention
# -------------------------------------------------- #
'''
embed_size: 每个单词用多少长度的向量来表示
heads: 多头注意力的heads个数
'''
class selfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(selfattention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        # 每个head的处理的特征个数
        self.head_dim = embed_size // heads

        # 如果不能整除就报错
        assert (self.head_dim * self.heads == self.embed_size), 'embed_size should be divided by heads'

        # 三个全连接分别计算qkv
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 输出层
        self.fc_out = nn.Linear(self.head_dim * self.heads, embed_size)

    # 前向传播 qkv.shape==[b,seq_len,embed_size]
    def forward(self, values, keys, query, mask):

        # 获取batch
        N = query.shape[0]

        # 获取每个句子有多少个单词
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 维度调整 [b,seq_len,embed_size] ==> [b,seq_len,heads,head_dim]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 对原始输入数据计算q、k、v
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 爱因斯坦简记法，用于张量矩阵运算，q和k的转置矩阵相乘
        # queries.shape = [N, query_len, self.heads, self.head_dim]
        # keys.shape = [N, keys_len, self.heads, self.head_dim]
        # energy.shape = [N, heads, query_len, keys_len]
        energy = torch.einsum('nqhd, nkhd -> nhqk', [queries, keys])

        # 是否使用mask遮挡t时刻以后的所有q、k
        if mask is not None:
            # 将mask中所有为0的位置的元素，在energy中对应位置都置为 －1*10^10
            energy = energy.masked_fill(mask == 0, torch.tensor(-1e10))

        # 根据公式计算attention, 在最后一个维度上计算softmax
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # 爱因斯坦简记法矩阵元素，其中query_len == keys_len == value_len
        # attention.shape = [N, heads, query_len, keys_len]
        # values.shape = [N, value_len, heads, head_dim]
        # out.shape = [N, query_len, heads, head_dim]
        out = torch.einsum('nhql, nlhd -> nqhd', [attention, values])

        # 维度调整 [N, query_len, heads, head_dim] ==> [N, query_len, heads*head_dim]
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # 全连接，shape不变
        out = self.fc_out(out)
        return out


# -------------------------------------------------- #
# （2）multi_head_attention + FFN
# -------------------------------------------------- #
'''
embed_size: 每个知识用多少长度的向量来表示
heads: 多头注意力的heas个数
dropout: 杀死神经元的概率
forward_expansion:  在FFN中第一个全连接上升特征数的倍数
'''
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        # 实例化自注意力模块
        self.attention = selfattention(embed_size, heads)

        # muti_head之后的layernorm
        self.norm1 = nn.LayerNorm(embed_size)
        # FFN之后的layernorm
        self.norm2 = nn.LayerNorm(embed_size)

        # 构建FFN前馈型神经网络
        self.feed_forward = nn.Sequential(
            # 第一个全连接层上升特征个数
            nn.Linear(embed_size, embed_size * forward_expansion),
            # relu激活
            nn.ReLU(),
            # 第二个全连接下降特征个数
            nn.Linear(embed_size * forward_expansion, embed_size)
        )

        # dropout层随机杀死神经元
        self.dropout = nn.Dropout(dropout)

    # 前向传播, qkv.shape==[b,seq_len,embed_size]
    def forward(self, value, key, query, mask):
        # 计算muti_head_attention
        attention = self.attention(value, key, query, mask)
        # 输入和输出做残差连接
        x = query + attention
        # layernorm标准化
        x = self.norm1(x)
        # dropout
        x = self.dropout(x)

        # FFN
        ffn = self.feed_forward(x)
        # 残差连接输入和输出
        forward = ffn + x
        # layernorm + dropout
        out = self.dropout(self.norm2(forward))

        return out


# -------------------------------------------------- #
# （3）RBF_Attention
# -------------------------------------------------- #
'''
key_dim: 值的维度
query_dim: 查询向量的维度
gamma: 高斯函数的宽度,过大的gamma会使核函数变得非常尖锐，只有非常近的样本对会有较高的相似度；而过小的gamma会使得核函数变得平坦，几乎所有的样本对都具有相似的高相似度。
lambda_reg：正则化项的权重超参数
'''
class RBF_Attention(nn.Module):
    def __init__(self, key_dim, query_dim, gamma, lambda_reg):
        super(RBF_Attention, self).__init__()
        # 定义用于生成注意力分数的线性层
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.epsilon = 1e-6

        '''
        self.key_layer = nn.Sequential(
            # relu激活
            nn.tanh(),
            # 第二个全连接下降特征个数
            nn.Linear(query_dim, key_dim, bias=False)  # 将查询映射到键的空间
        )
        '''
        self.key_layer = nn.Linear(query_dim, key_dim, bias=False)  # 将查询映射到键的空间

    def forward(self, query1, query2, keys1, keys2, values1, values2):
        """
        :param query1，query2: 查询张量，形状 [batch_size, query_dim]
        :param keys1，keys2: 键张量，形状 [batch_size, seq_len, key_dim]
        :param values1，values2: 值张量，形状 [batch_size, seq_len, value_dim]
        :return: 加权的值，形状 [batch_size, value_dim]
        """

        # 转换键
        transformed_query1 = self.key_layer(query1)  # [batch_size, query_dim]
        transformed_query2 = self.key_layer(query2)  # [batch_size, query_dim]
        transformed_query1 = F.normalize(transformed_query1, p=2, dim=-1)
        transformed_query2 = F.normalize(transformed_query2, p=2, dim=-1)
        keys1 = F.normalize(keys1, p=2, dim=-1)
        keys2 = F.normalize(keys2, p=2, dim=-1)

        # 计算查询和所有键之间的得分
        # 使用 batch matrix multiplication, 得分形状为 [batch_size, 1, seq_len]
        scores1 = torch.bmm(transformed_query1.unsqueeze(1), keys1.transpose(1, 2))
        scores2 = torch.bmm(transformed_query2.unsqueeze(1), keys2.transpose(1, 2))

        # 使用 softmax 获取注意力权重
        attn_weights1 = F.softmax(scores1-torch.max(scores1), dim=-1)  # 归一化得分 [batch_size, 1, seq_len]
        attn_weights2 = F.softmax(scores2-torch.max(scores2), dim=-1)  # 归一化得分 [batch_size, 1, seq_len]

        # 计算权重矩阵 attn_weights
        attn_weights = attn_weights2 * attn_weights1.permute(0, 2, 1)
        #attn_weights = torch.sigmoid(attn_weights)
        #print(attn_weights.shape)
        #print(attn_weights)
        # 计算加权的值
        loss = self.compute_loss(values1, values2, attn_weights)


        return attn_weights1, attn_weights2, loss

    def rbf_kernel(self, x1, x2):
        """计算批处理数据的RBF核"""
        # 计算每个向量对的欧氏距离，结果为[batch_size, seq_len, seq_len]
        dist = torch.norm(x1.unsqueeze(2) - x2.unsqueeze(1), dim=3)
        return torch.exp(-self.gamma * dist ** 2)

    def compute_loss(self, A, B, W):
        kernel_matrix = self.rbf_kernel(A, B)
        #print(kernel_matrix)
        # 归一化
        #kernel_matrix = 0.001 + (kernel_matrix - kernel_matrix.min()) * (1 - 0.001) / (kernel_matrix.max() - kernel_matrix.min())
        #kernel_matrix = torch.softmax(kernel_matrix, dim=-1)
        #print(A)
        #print(B)
        #print(W)
        #print(kernel_matrix)

        # 相似度奖励和不相似度惩罚
        sim_reward = W * kernel_matrix
        total_sim_reward = sim_reward.sum()
        #sim_reward = torch.softmax(sim_reward, dim=-1)
        #print(sim_reward)

        sim_penalty = W * (1 - kernel_matrix)
        total_sim_penalty = self.lambda_reg * sim_penalty.sum()
        #sim_penalty = torch.softmax(sim_penalty, dim=-1)
        #print(sim_penalty)

        total_sim = total_sim_reward - total_sim_penalty + max(total_sim_reward, abs(total_sim_penalty))

        #print(sim_reward.sum())
        #print(sim_penalty.sum())
        #print(self.lambda_reg * sim_penalty.sum())
        #print(total_sim)
        #print(1 / (total_sim ** 2 + self.epsilon))
        #total_sim = torch.exp(total_sim)
        #print(total_sim)

        return 1 / (total_sim ** 2 + self.epsilon)


# -------------------------------------------------- #
# （4）encoder
# -------------------------------------------------- #
'''
numOfEntity: 1+(一共有多少个实体)
numOfRelation: 1+(一共有多少个关系)
numOfTimestamp: 一共有多少个时间戳 
numOfTime_domain： 一共有多少个时间域 
Dimension：实体，关系，时间戳以及时间域的维度
num_layers: 堆叠多少层TransformerBlock
heads: 多头注意力的heads个数
drop: 在muti_head_atten和FFN之后的dropout层杀死神经元的概率
forward_expansion:  在FFN中第一个全连接上升特征数的倍数
gamma: 高斯函数的宽度,过大的gamma会使核函数变得非常尖锐，只有非常近的样本对会有较高的相似度；而过小的gamma会使得核函数变得平坦，几乎所有的样本对都具有相似的高相似度。
lambda_reg：正则化项的权重超参数
'''

class Encoder(nn.Module):
    def __init__(self, numOfEntity, numOfRelation, numOfTimestamp, numOfTime_domain, Dimension,
                 num_layers, heads, dropout, forward_expansion, gamma, lambda_reg):
        super(Encoder, self).__init__()

        self.numOfEntity = numOfEntity

        # 将每个实体用Dimension维的向量来表示
        self.Entity_embedding = nn.Embedding(numOfEntity, Dimension)
        self.Entity_embedding.weight.data = F.normalize(self.Entity_embedding.weight.data, 2, 1)

        # 将每个关系用Dimension维的向量来表示
        self.Relation_embedding = nn.Embedding(numOfRelation, Dimension)
        self.Relation_embedding.weight.data = F.normalize(self.Relation_embedding.weight.data, 2, 1)

        # 将每个时间戳用Dimension维的向量来表示
        self.Timestamp_embedding = nn.Embedding(numOfTimestamp, Dimension)
        self.Timestamp_embedding.weight.data = F.normalize(self.Timestamp_embedding.weight.data, 2, 1)

        # 将每个时间域用Dimension维的向量来表示
        self.Time_domain_embedding = nn.Embedding(numOfTime_domain, Dimension)
        self.Time_domain_embedding.weight.data = F.normalize(self.Time_domain_embedding.weight.data, 2, 1)

        self.dropout = nn.Dropout(dropout)

        # 将多个TransformerBlock保存在列表中
        self.layers = nn.ModuleList(
            [TransformerBlock(Dimension, heads, dropout, forward_expansion)
             for _ in range(num_layers)]
        )

        # 构建RBF_Attention层
        self.RBF_Attention = RBF_Attention(Dimension, Dimension, gamma, lambda_reg)

    # 前向传播Head_specific_knowledge_sequence.shape = Tail_specific_knowledge_sequence.shape = [batch, seq_len, kno_len]，x_Relation = [batch, relation_len]
    # seq_len为知识序列的长度（ = numOfTimestamp），kno_len为单个知识的长度（ = 5），relation_len为关系索引（ = 1）。
    def forward(self, Head_specific_knowledge_sequence, Tail_specific_knowledge_sequence, Relation, Head, Tail, mask):
        # 知识编码，将Head和Tail的过去知识序列经过Entity_embedding，Relation_embedding，Timestamp_embedding，Time_domain_embedding进行编码，然后相加，得到的shape=[batch, seq_len, Dimension]
        #print(Head_specific_knowledge_sequence)
        #print(Tail_specific_knowledge_sequence)
        Embedding_Head_specific_knowledge_sequence = self.Entity_embedding(Head_specific_knowledge_sequence[:, :, 0]) + self.Relation_embedding(Head_specific_knowledge_sequence[:, :, 1]) + self.Entity_embedding(Head_specific_knowledge_sequence[:, :, 2]) + self.Timestamp_embedding(Head_specific_knowledge_sequence[:, :, 3]) + self.Time_domain_embedding(Head_specific_knowledge_sequence[:, :, 4])
        Embedding_Tail_specific_knowledge_sequence = self.Entity_embedding(Tail_specific_knowledge_sequence[:, :, 0]) + self.Relation_embedding(Tail_specific_knowledge_sequence[:, :, 1]) + self.Entity_embedding(Tail_specific_knowledge_sequence[:, :, 2]) + self.Timestamp_embedding(Tail_specific_knowledge_sequence[:, :, 3]) + self.Time_domain_embedding(Tail_specific_knowledge_sequence[:, :, 4])

        # 将关系经过Relation_embedding，注意这里的关系不经过Transformer
        Embedding_Relation = self.Relation_embedding(Relation)

        # 将实体经过Entity_embedding，注意这里的实体不经过Transformer
        Embedding_Head = self.Entity_embedding(Head)
        Embedding_Tail = self.Entity_embedding(Tail)

        # dropout层
        #Embedding_Head = self.dropout(Embedding_Head)
        #Embedding_Tail = self.dropout(Embedding_Tail)

        #print(Embedding_Head_specific_knowledge_sequence)
        #print(Embedding_Tail_specific_knowledge_sequence)

        Out_Embedding_Head = Embedding_Head_specific_knowledge_sequence
        Out_Embedding_Tail = Embedding_Tail_specific_knowledge_sequence
        # 堆叠多个TransformerBlock层
        for layer in self.layers:
            Out_Embedding_Head = layer(Out_Embedding_Head, Out_Embedding_Head, Out_Embedding_Head, mask)
            Out_Embedding_Tail = layer(Out_Embedding_Tail, Out_Embedding_Tail, Out_Embedding_Tail, mask)

        #print(Out_Embedding_Head)
        #print(Out_Embedding_Tail)
        # 经过RBF_Attention，计算对不同时间戳下知识的关注度，并计算RBF核函数值
        attn_weights1, attn_weights12, loss = self.RBF_Attention(Embedding_Head + Embedding_Relation, Embedding_Tail + Embedding_Relation, Out_Embedding_Head, Out_Embedding_Tail, Out_Embedding_Head, Out_Embedding_Tail)


        return attn_weights1, attn_weights1, loss












