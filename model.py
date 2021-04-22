# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the License
# located at the root directory.
# ---------------------------------------------------------------

from copy import deepcopy
from typing import Union

import torch
from torch.nn import functional as F
from torch import nn
from torch.nn.functional import triplet_margin_loss

from data import CompDataFromDict
from useful_utils import ns_profiling_label
from params import CommandlineArgs
from ATTOP.models.models import MLP as ATTOP_MLP

def get_model(args: CommandlineArgs, dataset: CompDataFromDict):
    cfg = args.model
    if cfg.E_num_common_hidden==0:
        ECommon = nn.Sequential()
        ECommon.output_shape = lambda: (None, dataset.input_shape[0])
    else:
        ECommon = ATTOP_MLP(dataset.input_shape[0], cfg.h_dim, num_layers=cfg.E_num_common_hidden - 1, relu=True, bias=True).to(args.device)
        ECommon.output_shape = lambda: (None, cfg.h_dim)


    if not cfg.VisProd:
        h_A = Label_MLP_embed(dataset.num_attrs, cfg.h_dim,
                              num_layers=cfg.nlayers_label).to(args.device)
        h_O = Label_MLP_embed(dataset.num_objs, cfg.h_dim,
                              num_layers=cfg.nlayers_label).to(args.device)
        g1_emb_to_hidden_feat = ATTOP_MLP(2 * cfg.h_dim, cfg.h_dim,
                                          num_layers=cfg.nlayers_joint_ao).to(args.device)
        g2_feat_to_image_feat = ATTOP_MLP(cfg.h_dim, dataset.input_shape[0], num_layers=cfg.nlayers_joint_ao
                                          ).to(args.device)

    g_inv_O = MLP_Encoder(ECommon.output_shape()[1], h_dim=cfg.h_dim,
                          E_num_common_hidden=cfg.E_num_hidden - cfg.E_num_common_hidden,
                          mlp_activation='leaky_relu', BN=True).to(args.device)  # category
    g_inv_A = MLP_Encoder(ECommon.output_shape()[1], h_dim=cfg.h_dim,
                          E_num_common_hidden=cfg.E_num_hidden - cfg.E_num_common_hidden,
                          mlp_activation='leaky_relu', BN=True).to(args.device)  # category

    emb_cf_O = EmbeddingClassifier(h_O, image_feat_dim=g_inv_O.h_dim, device=args.device).to(args.device)
    # redundant historic call - just to make sure random-number-gen is kept aligned with original codebase
    _ = EmbeddingClassifier(h_A, image_feat_dim=g_inv_A.h_dim, device=args.device).to(args.device)

    emb_cf_A = EmbeddingClassifier(h_A, image_feat_dim=g_inv_A.h_dim, device=args.device).to(args.device)
    # redundant historic call - just to make sure random-number-gen is kept aligned with original codebase
    _ = EmbeddingClassifier(h_O, image_feat_dim=g_inv_O.h_dim, device=args.device).to(args.device)

    model = CompModel(ECommon, g_inv_O, g_inv_A, emb_cf_O, emb_cf_A, h_O, h_A, g1_emb_to_hidden_feat, g2_feat_to_image_feat,
                      args, dataset).to(args.device)

    return model


def nll_sum_loss(attr_logits, obj_logits, attr_gt, obj_gt, nll_loss_funcs):
    nll_loss_attr = nll_loss_funcs.y2(attr_logits, attr_gt)
    nll_loss_obj = nll_loss_funcs.y1(obj_logits, obj_gt)
    return nll_loss_attr + nll_loss_obj


def get_activation_layer(activation):
    return dict(leaky_relu=nn.LeakyReLU(),
                relu=nn.ReLU(),
                )[activation]

class MLP_block(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_activation='leaky_relu', BN=True):
        super(MLP_block, self).__init__()
        layers_list = []
        layers_list += [nn.Linear(input_dim, output_dim)]
        if BN:
            layers_list += [nn.BatchNorm1d(num_features=output_dim)]
        layers_list += [get_activation_layer(mlp_activation)]
        self.NN = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.NN(x)


class Label_MLP_embed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_layers=0):
        super(Label_MLP_embed, self).__init__()
        self.lin_emb = nn.Embedding(num_embeddings, embedding_dim)
        layers_list = []
        layers_list += [self.lin_emb]
        if num_layers > 0:
            layers_list += [ATTOP_MLP(embedding_dim, embedding_dim, num_layers=num_layers)]
        self.NN = nn.Sequential(*layers_list)
        self.embedding_dim = embedding_dim

    def forward(self, tokens):
        output = self.NN(tokens)
        return output



class MLP_Encoder(nn.Module):
    def __init__(self, input_dim, h_dim=150, E_num_common_hidden=1, mlp_activation='leaky_relu', BN=True,
                 linear_last_layer=False, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        layers_list = []
        prev_dim = input_dim
        for n in range(E_num_common_hidden):
            if n==0:
                layers_list += [MLP_block(prev_dim, h_dim, mlp_activation=mlp_activation, BN=BN)]
            else:
                layers_list += [MLP_block(prev_dim, h_dim, mlp_activation=mlp_activation, BN=BN)]
            prev_dim = h_dim


        if linear_last_layer:
            layers_list += [nn.Linear(prev_dim, prev_dim)]

        self._output_shape = prev_dim
        self.layers_list = layers_list
        self.NN = nn.Sequential(*layers_list)


    def forward(self, x):
        return self.NN(x)

    def output_shape(self):
        return (None, self._output_shape)


class EmbeddingClassifier(nn.Module):
    def __init__(self, embeddings, image_feat_dim, device):
        super().__init__()
        self.num_emb_class, _ = list(embeddings.parameters())[0].shape
        self.image_feat_dim = image_feat_dim
        self.device = device

        self.embeddings = embeddings
        self.emb_alignment = nn.Linear(image_feat_dim, self.embeddings.embedding_dim)
    def forward(self, feat):
        feat = self.emb_alignment(feat)
        emb_per_class = self.embeddings(torch.arange(0, self.num_emb_class, dtype=int).to(self.device)).T
        out = -((feat[:, :, None] - emb_per_class) ** 2).sum(1)
        return out


def LinearSoftmaxLogits(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.LogSoftmax())


class CompModel(nn.Module):
    def __init__(self, ECommon, g_inv_O, g_inv_A, emb_cf_O, emb_cf_A, h_O, h_A, g1_emb_to_hidden_feat, g2_feat_to_image_feat,
                 args: CommandlineArgs, dataset):
        """
        Input:
            E: encoder
            M: classifier
            D: discriminator
            alpha: weighting parameter of label classifier and domain classifier
            num_classes: the number of classes
         """
        super(CompModel, self).__init__()
        model_cfg = args.model
        self.args = args
        self.is_mutual = True
        self.ECommon = ECommon
        self.g_inv_O = g_inv_O
        self.g_inv_A = g_inv_A
        self.emb_cf_O = emb_cf_O
        self.emb_cf_A = emb_cf_A
        self.h_O = h_O
        self.h_A = h_A
        self.g1_emb_to_hidden_feat = g1_emb_to_hidden_feat
        self.g2_feat_to_image_feat = g2_feat_to_image_feat
        self.loader = None
        self.num_objs = dataset.num_objs
        self.num_attrs = dataset.num_attrs
        self.attrs_idxs, self.objs_idxs = self._get_ao_outerprod_idxs()
        self.last_feature_common = None

        if not model_cfg.VisProd:
            self.obj_inv_core_logits = LinearSoftmaxLogits(model_cfg.h_dim, self.num_objs).to(args.device)
            self.attr_inv_core_logits = LinearSoftmaxLogits(model_cfg.h_dim, self.num_attrs).to(args.device)

            self.obj_inv_g_hidden_logits = LinearSoftmaxLogits(model_cfg.h_dim, self.num_objs).to(args.device)
            self.attr_inv_g_hidden_logits = LinearSoftmaxLogits(model_cfg.h_dim, self.num_attrs).to(args.device)

            self.obj_inv_g_imgfeat_logits = LinearSoftmaxLogits(dataset.input_shape[0], self.num_objs).to(args.device)
            self.attr_inv_g_imgfeat_logits = LinearSoftmaxLogits(dataset.input_shape[0], self.num_attrs).to(args.device)

            self.device = args.device
            # check (rename)
            self.mu_disjoint = args.train.mu_disjoint
            self.mu_ao_emb = args.train.mu_ao_emb
            self.mu_img_feat = args.train.mu_img_feat

    def encode(self, input_data, freeze_class1=False, freeze_class2=False):
        feature_common = self.ECommon(input_data)
        self.last_feature_common = feature_common
        if feature_common is input_data:
            self.last_feature_common = None

        if freeze_class1:
            with torch.no_grad():
                feature1 = self.g_inv_O(feature_common).detach()
        else:
            feature1 = self.g_inv_O(feature_common)


        if freeze_class2:
            with torch.no_grad():
                feature2 = self.g_inv_A(feature_common).detach()
        else:
            feature2 = self.g_inv_A(feature_common)

        return feature1, feature2, feature_common

    def forward(self, input_data,
                freeze_class1=False, freeze_class2=False):

        ### init and definitions
        freeze_class = freeze_class1, freeze_class2
        classifiers = self.emb_cf_O, self.emb_cf_A
        class_outputs = [None, None]

        def set_grad_disabled(condition):
            return torch.set_grad_enabled(not condition)
        ### end init and definitions

        feature1, feature2, feature_common = self.encode(input_data, freeze_class1, freeze_class2)

        for m, feature in enumerate([feature1, feature2]):
            with set_grad_disabled(freeze_class[m]):
                class_outputs[m] = classifiers[m](feature)
                if freeze_class[m]:
                    class_outputs[m] = class_outputs[m].detach()

        joint_output = (class_outputs[0][..., None] + class_outputs[1][..., None, :])
        joint_output = torch.flatten(joint_output[:, :], start_dim=1)  # flatten

        # inference
        if not self.training and not self.args.model.VisProd:
            joint_output = self.mu_disjoint * joint_output

            if self.mu_img_feat > 0 or self.mu_ao_emb > 0:
                flattened_ao_emb_joint_scores, flattened_img_emb_scores = \
                    self.get_joint_embed_classification_scores(self.attrs_idxs, self.objs_idxs,
                                                               self.last_feature_common, input_data)

                joint_output += self.mu_ao_emb * flattened_ao_emb_joint_scores
                joint_output += self.mu_img_feat * flattened_img_emb_scores

            scores_emb = joint_output.view((-1, self.num_objs, self.num_attrs))
            class_outputs[0] = scores_emb.max(axis=2)[0].detach()
            class_outputs[1] = scores_emb.max(axis=1)[0].detach() # obj, attr

        return tuple(class_outputs + [feature1, feature2, joint_output])

    def eval_pair_embed_losses(self, args: CommandlineArgs, img_feat, img_hidden_emb, attr_labels, obj_labels,
                               neg_attr_labels, neg_obj_labels, nll_loss_funcs):
        device = args.device

        with ns_profiling_label('labels_to_embeddings'):
            h_A_pos, h_O_pos, g_hidden_pos, g_img_pos = self.labels_to_embeddings(attr_labels, obj_labels)
            _, _, g_hidden_neg, g_img_neg = self.labels_to_embeddings(neg_attr_labels, neg_obj_labels)

        tloss_g_imgfeat = torch.tensor(0.).to(device)
        if args.train.lambda_feat > 0:
            with ns_profiling_label('tloss_g_imgfeat'):
                tloss_g_imgfeat = triplet_margin_loss(img_feat, g_img_pos, g_img_neg,
                                                        margin=args.train.triplet_loss_margin)

        tloss_g_hidden = torch.tensor(0.).to(device)
        if args.train.lambda_ao_emb > 0:
            with ns_profiling_label('tloss_g_hidden'):
                tloss_g_hidden = triplet_margin_loss(img_hidden_emb, g_hidden_pos, g_hidden_neg,
                                                       margin=args.train.triplet_loss_margin)


        # Loss_invert terms
        loss_inv_core = torch.tensor(0.).to(device)
        if args.train.lambda_aux_disjoint > 0:  # check hp name
            with ns_profiling_label('loss_inv_core'):
                loss_inv_core = nll_sum_loss(self.attr_inv_core_logits(h_A_pos),
                                             self.obj_inv_core_logits(h_O_pos),
                                             attr_labels, obj_labels, nll_loss_funcs)

        loss_inv_g_imgfeat = torch.tensor(0.).to(device)
        if args.train.lambda_aux_img > 0:  # check hp name
            with ns_profiling_label('loss_inv_g_imgfeat'):
                loss_inv_g_imgfeat = nll_sum_loss(self.attr_inv_g_imgfeat_logits(g_img_pos),
                                                  self.obj_inv_g_imgfeat_logits(g_img_pos),
                                                  attr_labels, obj_labels, nll_loss_funcs)

        loss_inv_g_hidden = torch.tensor(0.).to(device)
        if args.train.lambda_aux > 0:  # check hp name
            with ns_profiling_label('loss_inv_g_hidden'):
                loss_inv_g_hidden = nll_sum_loss(self.attr_inv_g_hidden_logits(g_hidden_pos),
                                                 self.obj_inv_g_hidden_logits(g_hidden_pos),
                                                 attr_labels, obj_labels, nll_loss_funcs)


        return tloss_g_hidden, tloss_g_imgfeat, loss_inv_core, loss_inv_g_hidden, loss_inv_g_imgfeat

    def labels_to_embeddings(self, attr_labels, obj_labels):
        """
                        h_A
                        |
        attr_labels -> h_A ->
                         > g1_emb_to_hidden_feat ->  g2_feat_to_image_feat ->
        obj_labels  -> h_O ->                    |                         |
                        |                   g_hidden                  g_img
                        h_O

        """
        h_A = self.h_A(attr_labels)
        h_O = self.h_O(obj_labels)

        g_hidden = self.g1_emb_to_hidden_feat(torch.cat((h_A, h_O), dim=1))
        g_img = self.g2_feat_to_image_feat(g_hidden)

        return h_A, h_O, g_hidden, g_img

    def get_joint_embed_classification_scores(self, attrs, objs, common_emb_feat, img_feat):
        _, _, g_hidden, g_img = self.labels_to_embeddings(attrs, objs)
        vec_dist_img_emb = ((img_feat[:, :, None] - g_img.T[None, :, :]))
        flattened_img_emb_scores = -((vec_dist_img_emb ** 2).sum(1))

        if common_emb_feat is not None:
            vec_dist_joint_ao_emb = ((common_emb_feat[:, :, None] - g_hidden.T[None, :, :]))
            flattened_joint_scores = -((vec_dist_joint_ao_emb ** 2).sum(1))
        else:
            flattened_joint_scores = 0*flattened_img_emb_scores.detach()

        return flattened_joint_scores, flattened_img_emb_scores

    def _get_ao_outerprod_idxs(self):
        device = self.args.device
        outerprod_pairs = torch.cartesian_prod(torch.arange(0, self.num_objs, device=device),
                                               torch.arange(0, self.num_attrs, device=device))
        objs_idxs = outerprod_pairs[:, 0]
        attrs_idxs = outerprod_pairs[:, 1]
        return attrs_idxs, objs_idxs
