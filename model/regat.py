"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""

import torch
import torch.nn as nn
from model.fusion import BAN, BUTD, MuTAN, AttPooling
from model.language_model import WordEmbedding, QuestionEmbedding,\
                                 QuestionSelfAttention
from model.relation_encoder import ImplicitRelationEncoder,\
                                   ExplicitRelationEncoder
from model.classifier import SimpleClassifier


class ReGAT(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, q_att, v_relation,
                 joint_embedding, classifier, glimpse, fusion, relation_type):
        super(ReGAT, self).__init__()
        self.name = "ReGAT_%s_%s" % (relation_type, fusion)
        self.relation_type = relation_type
        self.fusion = fusion
        self.dataset = dataset
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.v_relation = v_relation  # 3 GAT
        self.joint_embedding = joint_embedding  # BAN, BUTD, MUTAN
        self.classifier = classifier  # MLP

    def forward(self, v, b, q, implicit_pos_emb, sem_adj_matrix,
                spa_adj_matrix):
        """Forward
        v: [batch, num_objs, obj_dim] visual features
        b: [batch, num_objs, b_dim] bounding box
        q: [batch_size, seq_length] question embedding
        pos: [batch_size, num_objs, nongt_dim, emb_dim]
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        implicit_pos_emb: [B, 20, 60, 64]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)  # [64, 14, 600]
        q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]  [64, 14, 1024]
        q_emb_self_att = self.q_att(q_emb_seq)  # [B, 1024]

        # [batch_size, num_rois, out_dim]
        if self.relation_type == "semantic":
            v_emb = self.v_relation.forward(v, sem_adj_matrix, q_emb_self_att)
        elif self.relation_type == "spatial":
            v_emb = self.v_relation.forward(v, spa_adj_matrix, q_emb_self_att)
        else:  # implicit & att pooling
            v_emb = self.v_relation.forward(v, implicit_pos_emb,
                                            q_emb_self_att)  # [64, 36, 1024]

        if self.fusion == "ban":
            joint_emb, att = self.joint_embedding(v_emb, q_emb_seq, b)
        elif self.fusion == "butd":
            q_emb = self.q_emb(w_emb)  # [batch, q_dim]
            joint_emb, att = self.joint_embedding(v_emb, q_emb)
        else:  # mutan & att pooling
            # [64, 3129], [64, 2048]
            joint_emb, att = self.joint_embedding(v_emb, q_emb_self_att)

        if self.classifier:
            logits = self.classifier(joint_emb)
        else:
            logits = joint_emb
        return logits, att


def build_regat(dataset, args):
    args.num_classes = dataset.num_ans_candidates
    print("Building ReGAT model with %s relation and %s fusion method" %
          (args.relation_type, args.fusion))
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600,
                              args.num_hid, 1, False, .0)
    q_att = QuestionSelfAttention(args.num_hid, .2)

    if args.relation_type == "semantic":
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.sem_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    elif args.relation_type == "spatial":
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.spa_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    else:
        v_relation = ImplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
                        num_heads=args.num_heads, num_steps=args.num_steps,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)

    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dataset.num_ans_candidates, 0.5)
    gamma = 0
    if args.fusion == "ban":
        joint_embedding = BAN(args.relation_dim, args.num_hid, args.ban_gamma)
        gamma = args.ban_gamma
    elif args.fusion == "butd":
        joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)
    elif args.fusion == 'att_pooling':
        joint_embedding = AttPooling(args)
        classifier = None
    else:
        joint_embedding = MuTAN(args.relation_dim, args.num_hid,
                                dataset.num_ans_candidates, args.mutan_gamma)
        gamma = args.mutan_gamma
        classifier = None
    return ReGAT(dataset, w_emb, q_emb, q_att, v_relation, joint_embedding,
                 classifier, gamma, args.fusion, args.relation_type)
