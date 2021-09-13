# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, classification_report

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


logger = logging.getLogger(__name__)


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1_cdcp(preds, labels):
        #acc = simple_accuracy(preds, labels)
        #f1 = f1_score(y_true=labels, y_pred=preds)

        logger.info("LABELS: {} \nPREDS: {}".format(str(labels), str(preds)))
        f1_micro = f1_score(labels, preds, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], average='micro')
        f1_macro = f1_score(labels, preds, average='macro')
        f1_policy = f1_score(labels, preds, labels=[1, 2], average='micro')
        f1_value = f1_score(labels, preds, labels=[3, 4], average='micro')
        f1_testimony = f1_score(labels, preds, labels=[5, 6], average='micro')
        f1_fact = f1_score(labels, preds, labels=[7, 8], average='micro')
        f1_reference = f1_score(labels, preds, labels=[9, 10], average='micro')

        return {
            #"acc": acc,
            #"f1": f1,
            'eval_f1_micro': f1_micro,
            'eval_f1_macro': f1_macro,
            'f1_policy': f1_policy,
            'f1_value': f1_value,
            'f1_testimony': f1_testimony,
            'f1_fact': f1_fact,
            'f1_reference': f1_reference,
            #"acc_and_f1": (acc + f1) / 2,
        }

    def acc_and_f1(preds, labels):
        #acc = simple_accuracy(preds, labels)
        #f1 = f1_score(y_true=labels, y_pred=preds)
        f1_micro = f1_score(labels, preds, labels=[1, 2, 3, 4, 5], average='micro')
        f1_macro = f1_score(labels, preds, average='macro')
        f1_claim = f1_score(labels, preds, labels=[1,2], average='micro')
        f1_evidence = f1_score(labels, preds, labels=[3,4], average='micro')

        return {
            #"acc": acc,
            #"f1": f1,
            'eval_f1_micro': f1_micro,
            'eval_f1_macro': f1_macro,
            'f1_claim':f1_claim,
            'f1_evidence':f1_evidence,
            #"acc_and_f1": (acc + f1) / 2,
        }

    def f1_scores(y_pred, y_true):

        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        f1_none = f1_score(y_true, y_pred, labels=[0], average=None)[0]
        f1_reason = f1_score(y_true, y_pred, labels=[1], average=None)[0]
        f1_evidence = f1_score(y_true, y_pred, labels=[2], average=None)[0]

        clf_report = classification_report(y_true, y_pred)

        return {
            'eval_f1_micro': f1_micro,
            'eval_f1_macro': f1_macro,
            'eval_f1_none': f1_none,
            'eval_f1_reason': f1_reason,
            'eval_f1_evidence': f1_evidence,
            'clf_report': clf_report
        }

    def f1_scores_inv(y_pred, y_true):

        f1_reason_inv = f1_score(y_true, y_pred, labels=[2], average='macro')
        f1_evidence_inv = f1_score(y_true, y_pred, labels=[4], average='macro')

        replace = {0: 0,
                   1: 1,
                   2: 0,
                   3: 2,
                   4: 0}

        y_true = [replace[x] for x in y_true]
        y_pred = [replace[x] for x in y_pred]

        # f1_micro_filtered = f1_score(y_true, y_pred, labels=labelfilter, average='micro')
        # f1_macro_filtered = f1_score(y_true, y_pred, labels=labelfilter, average='macro')
        f1_micro = f1_score(y_true, y_pred, labels=[0, 1, 2], average='micro')
        f1_macro = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro')

        f1_none = f1_score(y_true, y_pred, labels=[0], average=None)[0]
        f1_reason = f1_score(y_true, y_pred, labels=[1], average=None)[0]
        f1_evidence = f1_score(y_true, y_pred, labels=[2], average=None)[0]

        clf_report = classification_report(y_true, y_pred)

        return {
            'eval_f1_micro': f1_micro,
            'eval_f1_macro': f1_macro,
            'eval_f1_none': f1_none,
            'eval_f1_reason': f1_reason,
            'eval_f1_evidence': f1_evidence,
            'eval_f1_r_inv': f1_reason_inv,
            'eval_f1_e_inv': f1_evidence_inv,
            'clf_report': clf_report
        }

    def f1_scores_components(y_true, y_pred, comp_set):

        f1_macro = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro')
        all_f1s = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None)

        return {
            comp_set + 'f1_macro': f1_macro,
            comp_set + 'f1_policy': all_f1s[0],
            comp_set + 'f1_fact': all_f1s[1],
            comp_set + 'f1_testimony': all_f1s[2],
            comp_set + 'f1_value': all_f1s[3],
            comp_set + 'f1_reference': all_f1s[4],
        }



    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def compute_confusion_matrix(task_name, y_pred, y_true):

        assert len(y_pred) == len(y_true)
        if task_name == "multichoice" or task_name == "relclass":
            return confusion_matrix(y_true, y_pred)
        else:
            raise KeyError(task_name)


    f1_scores_tasks = ["relclass", "cdcp_relclass", "cdcp_relclass_context", "cdcp_relclass_rbert",
                       "cdcp_relclass_rbert_jl", "cdcp_relclass_distance", "cdcp_relclass_distance_components",
                       "cdcp_relclass_resnet", "cdcp_relclass_resnet_jl", "multichoice", "outcomeclf"]

    f1_scores_tasks_inverse = ["cdcp_relclass_resnet_jl_inv", "cdcp_relclass_jl_inv"]

    def compute_metrics(task_name, y_pred, y_true):
        assert len(y_pred) == len(y_true)

        if task_name in f1_scores_tasks:
            return f1_scores(y_pred, y_true)
        elif task_name == "cdcp_seqtag":
            return acc_and_f1_cdcp(y_pred, y_true)
        elif task_name in f1_scores_tasks_inverse:
            return f1_scores_inv(y_pred, y_true)
        elif task_name == "MTL-source":
            return f1_scores_components(y_pred, y_true, 'source')
        elif task_name == "MTL-target":
            return f1_scores_components(y_pred, y_true, 'target')

        elif task_name == "seqtag":
            return acc_and_f1(y_pred, y_true)
        else:
            raise KeyError(task_name)

