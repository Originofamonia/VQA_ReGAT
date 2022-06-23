"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torchviz import make_dot

import utils
from model.position_emb import prepare_graph_variables


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    # loss = F.binary_cross_entropy_with_logits(
    #     logits, labels, reduction=reduction)
    # if reduction == "mean":
    #     loss = loss * labels.size(1)
    return loss


def compute_score_with_logits(logits, labels, device):
    # argmax
    logits = torch.max(logits, 1)[1].data
    logits = logits.view(-1, 1)
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, args, device=torch.device("cuda")):
    n = len(train_loader.dataset)
    lr_default = args.base_lr
    num_epochs = args.epochs
    lr_decay_epochs = range(args.lr_decay_start, num_epochs,
                            args.lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,
                            1.5 * lr_default, 2.0 * lr_default]

    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad,
                                          model.parameters()),
                                   lr=lr_default, betas=(0.9, 0.999), eps=1e-8,
                                   weight_decay=args.weight_decay)

    # logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    # best_eval_score = 0

    # print(model)
    print('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f,'
          % (lr_default, args.lr_decay_step,
             args.lr_decay_rate) + 'grad_clip=%.2f' % args.grad_clip)
    print('LR decay epochs: ' + ','.join(
        [str(i) for i in lr_decay_epochs]))
    last_eval_score, eval_score = 0, 0
    relation_type = train_loader.dataset.relation_type

    # visualize(model, eval_loader, device, args)
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader)
        total_norm, count_norm = 0, 0
        total_loss, train_score = 0, 0
        count, average_loss, att_entropy = 0, 0, 0
        t = time.time()
        if epoch < len(gradual_warmup_steps):
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = gradual_warmup_steps[epoch]
            print('gradual warmup lr: %.4f' %
                  optimizer.param_groups[-1]['lr'])
        elif (epoch in lr_decay_epochs or
              eval_score < last_eval_score and args.lr_decay_based_on_val):
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] *= args.lr_decay_rate
            print('decreased lr: %.4f' % optimizer.param_groups[-1]['lr'])
        else:
            print('lr: %.4f' % optimizer.param_groups[-1]['lr'])
        last_eval_score = eval_score

        mini_batch_count = 0
        batch_multiplier = args.grad_accu_steps
        for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix,
                sem_adj_matrix) in enumerate(pbar):
            batch_size = v.size(0)
            num_objects = v.size(1)
            if mini_batch_count == 0:
                optimizer.step()
                optimizer.zero_grad()
                mini_batch_count = batch_multiplier
            # spa_adj_matrix: [128, 1], sem_adj_matrix: [128, 1]
            v = v.to(device)  # v: [128, 36, 2048]
            norm_bb = norm_bb.to(device)  # [128, 36, 6], bb: [128, 36, 4]
            q = q.to(device)  # [128, 14]
            target = target.to(device)  # [128, 3129] 128 is because 2 gpus
            pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
                relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
                args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
                args.sem_label_num, device)
            pred, att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                              spa_adj_matrix)
            loss = instance_bce_with_logits(pred, target)

            loss = loss / batch_multiplier
            loss.backward()
            mini_batch_count -= 1
            total_norm += nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip)
            count_norm += 1
            batch_score = compute_score_with_logits(pred, target, device).sum()
            total_loss += loss.data.item() * batch_multiplier * v.size(0)
            train_score += batch_score

            if args.log_interval > 0:
                average_loss += loss.data.item() * batch_multiplier
                if model.fusion == "ban":
                    current_att_entropy = torch.sum(calc_entropy(att.data))
                    att_entropy += current_att_entropy / batch_size / att.size(1)
                count += 1
                if i % args.log_interval == 0:
                    att_entropy /= count
                    average_loss /= count
                    desc = "Step {} / {} (Epoch {}), ave_loss {:.3f}," \
                           " att_entropy {:.3f}".format(i, len(train_loader), epoch,
                                                        average_loss, att_entropy)
                    pbar.set_description(desc)
                    average_loss = 0
                    count = 0
                    att_entropy = 0

        total_loss /= n
        train_score = 100 * train_score / n
        print('Epoch %d, time: %.2f' % (epoch, time.time() - t))
        print('Train_loss: %.2f, norm: %.4f, score: %.2f'
              % (total_loss, total_norm / count_norm, train_score))
        if eval_loader is not None:
            eval_score, bound, entropy = evaluate(
                model, eval_loader, device, args)

            print('Eval score: %.2f (%.2f)'
                  % (100 * eval_score, 100 * bound))

            if entropy is not None:
                info = ''
                for i in range(entropy.size(0)):
                    info = info + ' %.2f' % entropy[i]
                print('Entropy: ' + info)
        # if (eval_loader is not None) \
        #         or (eval_loader is None and epoch >= args.saving_epoch):
        # logger.write("saving current model weights to folder")
        # model_path = os.path.join(args.output, 'model_%d.pth' % epoch)
        # opt = optimizer if args.save_optim else None
        # utils.save_model(model_path, model, epoch, opt)


@torch.no_grad()
def evaluate(model, dataloader, device, args):
    model.eval()
    relation_type = dataloader.dataset.relation_type
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = None
    if model.fusion == "ban":
        entropy = torch.Tensor(model.glimpse).zero_().to(device)
    pbar = tqdm(dataloader)

    for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix,
            sem_adj_matrix) in enumerate(pbar):
        num_objects = v.size(1)
        v = v.to(device)
        norm_bb = norm_bb.to(device)
        q = q.to(device)
        target = target.to(device)

        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num, device)
        pred, att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                          spa_adj_matrix)
        batch_score = compute_score_with_logits(
            pred, target, device).sum()
        score += batch_score
        upper_bound += (target.max(1)[0]).sum()
        num_data += pred.size(0)
        if att is not None and 0 < model.glimpse \
                and entropy is not None:
            entropy += calc_entropy(att.data)[:model.glimpse]

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    return score, upper_bound, entropy


@torch.no_grad()
def visualize(model, dataloader, device, args):
    model.eval()
    relation_type = dataloader.dataset.relation_type
    pbar = tqdm(dataloader)

    for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix,
            sem_adj_matrix) in enumerate(pbar):
        num_objects = v.size(1)
        v = v.to(device)
        norm_bb = norm_bb.to(device)
        q = q.to(device)
        # target = target.to(device)

        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num, device)
        # need to separate forward, make_dot can't visualize combined model.
        make_dot(model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                       spa_adj_matrix), params=dict(model.named_parameters()))

    model.train()
    # return score, upper_bound, entropy


def calc_entropy(att):
    # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p + eps).log()).sum(2).sum(0)  # g
