"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
from turtle import forward
import torch
import torch.nn as nn
from model.bilinear_attention import BiAttention
import torch.nn.functional as F
from model.fc import FCNet
from model.bc import BCNet
from model.counting import Counter
from torch.nn.utils.weight_norm import weight_norm
from block import fusions

"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is modified from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""


class BAN(nn.Module):
    def __init__(self, v_relation_dim, num_hid, gamma,
                 min_num_objects=10, use_counter=True):
        super(BAN, self).__init__()

        self.v_att = BiAttention(v_relation_dim, num_hid, num_hid, gamma)
        self.glimpse = gamma
        self.use_counter = use_counter
        b_net = []
        q_prj = []
        c_prj = []
        q_att = []
        v_prj = []

        for i in range(gamma):
            b_net.append(BCNet(v_relation_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            if self.use_counter:
                c_prj.append(FCNet([min_num_objects + 1, num_hid], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.q_att = nn.ModuleList(q_att)
        self.v_prj = nn.ModuleList(v_prj)
        if self.use_counter:
            self.c_prj = nn.ModuleList(c_prj)
            self.counter = Counter(min_num_objects)

    def forward(self, v_relation, q_emb, b):
        if self.use_counter:
            boxes = b[:, :, :4].transpose(1, 2)

        b_emb = [0] * self.glimpse
        # b x g x v x q
        att, att_logits = self.v_att.forward_all(v_relation, q_emb)

        for g in range(self.glimpse):
            # b x l x h
            b_emb[g] = self.b_net[g].forward_with_weights(
                                        v_relation, q_emb, att[:, g, :, :])
            # atten used for counting module
            atten, _ = att_logits[:, g, :, :].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

            if self.use_counter:
                embed = self.counter(boxes, atten)
                q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)
        joint_emb = q_emb.sum(1)
        return joint_emb, att

"""
This code is modified by Linjie Li from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
GNU General Public License v3.0
"""


class BUTD(nn.Module):
    def __init__(self, v_relation_dim, q_dim, num_hid, dropout=0.2):
        super(BUTD, self).__init__()
        self.v_proj = FCNet([v_relation_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = FCNet([num_hid, 1])
        self.q_net = FCNet([q_dim, num_hid])
        self.v_net = FCNet([v_relation_dim, num_hid])

    def forward(self, v_relation, q_emb):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        b: bounding box features, not used for this fusion method
        """
        logits = self.logits(v_relation, q_emb)
        att = nn.functional.softmax(logits, 1)
        v_emb = (att * v_relation).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_emb = q_repr * v_repr
        return joint_emb, att

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

"""
This code is modified by Linjie Li from Remi Cadene's repository.
https://github.com/Cadene/vqa.pytorch
"""


class MuTAN_Attention(nn.Module):
    def __init__(self, dim_v, dim_q, dim_out, method="Mutan", mlp_glimpses=0):
        super(MuTAN_Attention, self).__init__()
        self.mlp_glimpses = mlp_glimpses
        self.fusion = getattr(fusions, method)(
                        [dim_q, dim_v], dim_out, mm_dim=1200,
                        dropout_input=0.1)
        if self.mlp_glimpses > 0:
            self.linear0 = FCNet([dim_out, 512], '', 0)
            self.linear1 = FCNet([512, mlp_glimpses], '', 0)

    def forward(self, q, v):
        alpha = self.process_attention(q, v)

        if self.mlp_glimpses > 0:
            alpha = self.linear0(alpha)
            alpha = F.relu(alpha)
            alpha = self.linear1(alpha)

        alpha = F.softmax(alpha, dim=1)

        if alpha.size(2) > 1:  # nb_glimpses > 1
            alphas = torch.unbind(alpha, dim=2)
            v_outs = []
            for alpha in alphas:
                alpha = alpha.unsqueeze(2).expand_as(v)
                v_out = alpha*v
                v_out = v_out.sum(1)
                v_outs.append(v_out)
            v_out = torch.cat(v_outs, dim=1)
        else:
            alpha = alpha.expand_as(v)
            v_out = alpha*v
            v_out = v_out.sum(1)
        return v_out

    def process_attention(self, q, v):
        batch_size = q.size(0)
        n_regions = v.size(1)
        # q = q[:, None, :].expand(q.size(0), n_regions, q.size(1)).clone()
        q = torch.unsqueeze(q, dim=1).repeat(1, n_regions, 1)
        alpha = self.fusion([
            q.contiguous().view(batch_size * n_regions, -1),
            v.contiguous().view(batch_size * n_regions, -1)
        ])
        alpha = alpha.view(batch_size, n_regions, -1)
        return alpha


class MuTAN(nn.Module):
    def __init__(self, v_relation_dim, num_hid, num_ans_candidates, gamma):
        super(MuTAN, self).__init__()
        self.gamma = gamma
        self.attention = MuTAN_Attention(v_relation_dim, num_hid,
                                         dim_out=360, method="Mutan",
                                         mlp_glimpses=gamma)
        self.fusion = getattr(fusions, "Mutan")(
                        [num_hid, v_relation_dim*2], num_ans_candidates,
                        mm_dim=1200, dropout_input=0.1)

    def forward(self, v_relation, q_emb):
        # b: bounding box features, not used for this fusion method
        att = self.attention(q_emb, v_relation)
        logits = self.fusion([q_emb, att])
        return logits, att


class AttPooling(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        # no attflat_lan because regat's q is one vector
        self.attflat_img = AttFlat(args)
        self.proj_norm = nn.LayerNorm(args.flat_out_size)
        self.proj = nn.Linear(args.flat_out_size, args.num_classes)
    
    def forward(self, v, q):
        v_mask = self.make_mask(v)
        img_feat_flat, v_w = self.attflat_img(  # [B, 512]
            v, v_mask,  # change input to w/o mask
        )
        a = q + img_feat_flat
        a = self.proj_norm(a)  # [B, 512]
        logits = torch.sigmoid(self.proj(a))
        return logits, v_w
    
    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), 
            dim=-1) == 0).unsqueeze(1).unsqueeze(2)


class AttFlat(nn.Module):
    def __init__(self, args):
        super(AttFlat, self).__init__()
        self.opt = args

        self.mlp = MLP(
            in_size=args.num_hid,
            mid_size=args.flat_mlp_size,
            out_size=args.flat_glimpses,  # att heads
            dropout_rate=args.dropout_rate,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            args.hidden_size * args.flat_glimpses,
            args.flat_out_size
        )

    def forward(self, x, x_mask):
        att_w = self.mlp(x)  # [B, 60, 1]
        att_w = att_w.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att_w = F.softmax(att_w, dim=1)  # att weights

        att_list = []
        for i in range(self.opt.flat_glimpses):
            att_list.append(
                torch.sum(att_w[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)  # [B, 1024]
        x_atted = self.linear_merge(x_atted)

        return x_atted, att_w


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_rate
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_rate=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_rate=dropout_rate, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))
