"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import tools.compute_softscore
import itertools
import math

# TODO: merge dataset_cp_v2.py with dataset.py

COUNTING_ONLY = False


# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '')\
            .replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK
                # for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if answer is not None:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
        (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    # train, val
    if 'test' != name[:4]:
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, answer))
    # test2015
    else:
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, None))

    return entries


def _load_visualgenome(dataroot, name, img_id2val, label2ans, adaptive=True):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(dataroot,
                                 'visualGenome/question_answers.json')
    image_data_path = os.path.join(dataroot,
                                   'visualGenome/image_data.json')
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    cache_path = os.path.join(dataroot, 'cache', 'vg_%s%s_target.pkl' %
                              (name, '_adaptive' if adaptive else ''))

    if os.path.isfile(cache_path):
        entries = pickle.load(open(cache_path, 'rb'))
    else:
        entries = []
        ans2label = pickle.load(open(ans2label_path, 'rb'))
        vgq = json.load(open(question_path, 'r'))
        # 108,077 images
        _vgv = json.load(open(image_data_path, 'r'))
        vgv = {}
        for _v in _vgv:
            if _v['coco_id']:
                vgv[_v['image_id']] = _v['coco_id']
        # used image, used question, total question, out-of-split
        counts = [0, 0, 0, 0]
        for vg in vgq:
            coco_id = vgv.get(vg['id'], None)
            if coco_id is not None:
                counts[0] += 1
                img_idx = img_id2val.get(coco_id, None)
                if img_idx is None:
                    counts[3] += 1
                for q in vg['qas']:
                    counts[2] += 1
                    _answer = tools.compute_softscore.preprocess_answer(
                                q['answer'])
                    label = ans2label.get(_answer, None)
                    if label and img_idx:
                        counts[1] += 1
                        answer = {
                            'labels': [label],
                            'scores': [1.]}
                        entry = {
                            'question_id': q['qa_id'],
                            'image_id': coco_id,
                            'image': img_idx,
                            'question': q['question'],
                            'answer': answer}
                        if not COUNTING_ONLY \
                           or is_howmany(q['question'], answer, label2ans):
                            entries.append(entry)

        print('Loading VisualGenome %s' % name)
        print('\tUsed COCO images: %d/%d (%.4f)' %
              (counts[0], len(_vgv), counts[0]/len(_vgv)))
        print('\tOut-of-split COCO images: %d/%d (%.4f)' %
              (counts[3], counts[0], counts[3]/counts[0]))
        print('\tUsed VG questions: %d/%d (%.4f)' %
              (counts[1], counts[2], counts[1]/counts[2]))
        with open(cache_path, 'wb') as f:
            pickle.dump(entries, open(cache_path, 'wb'))

    return entries


def _find_coco_id(vgv, vgv_id):
    for v in vgv:
        if v['image_id'] == vgv_id:
            return v['coco_id']
    return None


class MIMICFeatureDataset(Dataset):
    def __init__(self, name, dictionary, relation_type, dataroot='data',
                 adaptive=False, pos_emb_dim=64, nongt_dim=36, pure_classification=True):
        super(MIMICFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015']

        ans2label_path = os.path.join(dataroot, 'mimic',
                                      'mimic_ans2label_full.pkl')
        label2ans_path = os.path.join(dataroot, 'mimic',
                                      'mimic_label2ans_full.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.relation_type = relation_type
        self.adaptive = adaptive
        self.pure_classification = pure_classification
        prefix = '36'
        if 'test' in name:
            prefix = '_36'

        h5_dataroot = dataroot+ '/mimic'

        h5_path = os.path.join(h5_dataroot, 'cmb_bbox_features_full.hdf5')

        print('loading features from h5 file %s' % h5_path)
        hf = h5py.File(h5_path, 'r')
        self.features = hf['image_features']
        self.normalized_bb = hf['spatial_features']
        self.bb = hf['image_bb']
        if "semantic_adj_matrix" in hf.keys() \
            and self.relation_type == "semantic":
            self.semantic_adj_matrix = hf['semantic_adj_matrix']
            print("Loaded semantic adj matrix from file...",
                    self.semantic_adj_matrix.shape)
        else:
            self.semantic_adj_matrix = None
            print("Setting semantic adj matrix to None...")
        if "image_adj_matrix" in hf.keys()\
            and self.relation_type == "spatial":
            self.spatial_adj_matrix = hf['image_adj_matrix']
            print("Loaded spatial adj matrix from file...",
                    self.spatial_adj_matrix.shape)
        else:
            self.spatial_adj_matrix = None
            print("Setting spatial adj matrix to None...")

        self.pos_boxes = None
        if self.adaptive:
            self.pos_boxes = hf['pos_boxes']
        
        if name == 'train':
                dataset_path = '/mimic/mimic_dataset_train_full.pkl'
        elif name == 'val':
            dataset_path = '/mimic/mimic_dataset_val_full.pkl'
        elif name == 'test':
            dataset_path = '/mimic/mimic_dataset_test_full.pkl'
        
        with open(dataroot + dataset_path, 'rb') as f:
            self.entries = pickle.load(f)  # qa
        
        self.tokenize()
        self.tensorize()
        self.nongt_dim = nongt_dim
        self.emb_dim = pos_emb_dim
        self.v_dim = self.features.shape[-1]
        self.s_dim = self.normalized_bb.shape[-1]

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            entry['q_token'] = self.sub_tokenize(entry['question'], max_length=max_length)
            # entry['a_token'] = self.sub_tokenize(self.enrich_answer(entry['answer']['answer']), max_length=max_length)

    def enrich_answer(self, anss):
        if anss[0] =="yes" or anss[0] == 'no':
            return anss[0]
        else:
            if len(anss) == 1:
                return 'an x-ray image contains ' + anss[0]
            else:
                ans = 'an x-ray image contains '
                for i in range(len(anss)):
                    if i == len(anss) -1:
                        ans += 'and ' + anss[i]
                    else:
                        ans += anss[i]+', '
            return ans

    def sub_tokenize(self, text, max_length=14):
        tokens = self.dictionary.tokenize(text, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad to the back of the sentence
            padding = [self.dictionary.padding_idx] * \
                      (max_length - len(tokens))
            tokens = tokens + padding
        utils.assert_eq(len(tokens), max_length)
        return tokens

    def tensorize(self):
        # self.features = torch.from_numpy(self.features)
        # self.normalized_bb = torch.from_numpy(self.normalized_bb)
        # self.bb = torch.from_numpy(self.bb)
        # if self.semantic_adj_matrix is not None:
        #     self.semantic_adj_matrix = torch.from_numpy(
        #                                 self.semantic_adj_matrix).double()
        # if self.spatial_adj_matrix is not None:
        #     self.spatial_adj_matrix = torch.from_numpy(
        #                                 self.spatial_adj_matrix).double()
        # if self.pos_boxes is not None:
        #     self.pos_boxes = torch.from_numpy(self.pos_boxes)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            # entry['a_token'] = torch.from_numpy(np.array(entry['a_token']))

            answer = entry['answer']
            if answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        # raw_question = entry["question"]
        image_id = entry["dicom_id"]

        question = entry['q_token']
        # question_id = entry['question_id']
        if self.spatial_adj_matrix is not None:
            spatial_adj_matrix = torch.from_numpy(self.spatial_adj_matrix[entry["image"]]).double()
        else:
            spatial_adj_matrix = torch.zeros(1).double()
        if self.semantic_adj_matrix is not None:
            semantic_adj_matrix = torch.from_numpy(self.semantic_adj_matrix[entry["image"]]).double()
        else:
            semantic_adj_matrix = torch.zeros(1).double()
        if not self.adaptive:  # adaptive or fixed number of regions
            # fixed number of bounding boxes
            features = torch.from_numpy(self.features[entry['image']])
            normalized_bb = torch.from_numpy(np.float32(self.normalized_bb[entry['image']]))
            bb = torch.from_numpy(self.bb[entry["image"]])
        else:
            features = self.features[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            normalized_bb = self.normalized_bb[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            bb = self.bb[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]

        answer = entry['answer']
        if answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            # target_ori = torch.zeros(self.num_ans_candidates) 
            # # there seems be no difference between target_ori and target2 right 
            # # now. but they are designed to be different for vqa/classification
            # if self.pure_classification:
            #     target2 = torch.zeros(self.num_ans_candidates)
            # else:
            #     target2 = torch.zeros(self.num_ans_candidates-1)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, normalized_bb, question, target,\
                image_id, bb, spatial_adj_matrix,\
                semantic_adj_matrix
        else:
            return features, normalized_bb, question, \
                image_id, bb, spatial_adj_matrix,\
                semantic_adj_matrix

    def __len__(self):
        return len(self.entries)


class VisualGenomeFeatureDataset(Dataset):
    def __init__(self, name, features, normalized_bb, bb,
                 spatial_adj_matrix, semantic_adj_matrix, dictionary,
                 relation_type, dataroot='data', adaptive=False,
                 pos_boxes=None, pos_emb_dim=64):
        super(VisualGenomeFeatureDataset, self).__init__()
        # do not use test split images!
        assert name in ['train', 'val']
        print('loading Visual Genome data %s' % name)
        ans2label_path = os.path.join(dataroot, 'cache',
                                      'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache',
                                      'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = pickle.load(
                open(os.path.join(dataroot, 'imgids/%s%s_imgid2idx.pkl' %
                                  (name, '' if self.adaptive else '36')),
                     'rb'))
        self.bb = bb
        self.features = features
        self.normalized_bb = normalized_bb
        self.spatial_adj_matrix = spatial_adj_matrix
        self.semantic_adj_matrix = semantic_adj_matrix

        if self.adaptive:
            self.pos_boxes = pos_boxes

        self.entries = _load_visualgenome(dataroot, name, self.img_id2idx,
                                          self.label2ans,
                                          adaptive=self.adaptive)
        self.tokenize()
        self.tensorize()
        self.emb_dim = pos_emb_dim
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.normalized_bb.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * \
                          (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        raw_question = entry["question"]
        image_id = entry["image_id"]
        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        if self.spatial_adj_matrix is not None:
            spatial_adj_matrix = self.spatial_adj_matrix[entry["image"]]
        else:
            spatial_adj_matrix = torch.zeros(1).double()
        if self.semantic_adj_matrix is not None:
            semantic_adj_matrix = self.semantic_adj_matrix[entry["image"]]
        else:
            semantic_adj_matrix = torch.zeros(1).double()
        if self.adaptive:
            features = self.features[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            normalized_bb = self.normalized_bb[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            bb = self.bb[self.pos_boxes[
                entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        else:
            features = self.features[entry['image']]
            normalized_bb = self.normalized_bb[entry['image']]
            bb = self.bb[entry['image']]

        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        return features, normalized_bb, question, target, raw_question,\
            image_id, bb, spatial_adj_matrix, semantic_adj_matrix

    def __len__(self):
        return len(self.entries)


def tfidf_from_questions(names, dictionary, dataroot='data',
                         target=['vqa', 'vg']):
    # rows, cols for uncoalesce sparse matrix
    inds = [[], []]
    df = dict()
    N = len(dictionary)

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0])
                inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1])
                inds[1].append(c[0])

    # VQA 2.0
    if 'vqa' in target:
        for name in names:
            assert name in ['train', 'val', 'test-dev2015', 'test2015']
            question_path = os.path.join(
                dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
                (name + '2014' if 'test' != name[:4] else name))
            questions = json.load(open(question_path))['questions']

            for question in questions:
                populate(inds, df, question['question'])

    # Visual Genome
    if 'vg' in target:
        question_path = os.path.join(dataroot, 'visualGenome',
                                     'question_answers.json')
        vgq = json.load(open(question_path, 'r'))
        for vg in vgq:
            for q in vg['qas']:
                populate(inds, df, q['question'])

    # TF-IDF
    vals = np.ones((len(inds[1])))
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds),
                                     torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = dataroot+'/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = utils.create_glove_embedding_init(
                        dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0),
          tfidf.size(1)))

    return tfidf, weights


# VisualGenome Train
#     Used COCO images: 51487/108077 (0.4764)
#     Out-of-split COCO images: 17464/51487 (0.3392)
#     Used VG questions: 325311/726932 (0.4475)

# VisualGenome Val
#     Used COCO images: 51487/108077 (0.4764)
#     Out-of-split COCO images: 34023/51487 (0.6608)
#     Used VG questions: 166409/726932 (0.2289)
