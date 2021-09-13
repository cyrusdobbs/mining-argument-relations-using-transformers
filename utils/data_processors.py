import csv
import json
import sys
import os
import collections
import torch
import logging
import time
import copy

import transformers
from torch.utils.data import TensorDataset
from transformers import BasicTokenizer
import logging

from preprocessing.pre_process_utils import CDCPArgumentationDoc

logger = logging.getLogger(__name__)

SEQUENCE_TAGGING = "sequence_tagging"
CLASSIFICATION = "classification"
REGRESSION = "regression"

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels

    @classmethod
    def truncate_seq_pair(cls, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class CDCPExample(InputExample):

    def __init__(self, guid, text_a, components, labels=None):
        super().__init__(guid, text_a, labels=labels)

        self.components = components


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

class RBERTInputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     logger=None,
                                     forSequenceTagging=False,
                                     min_seq_length=None,
                                     cls_token='[CLS]',
                                     sep_token_extra=False,
                                     sep_token='[SEP]'):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        max_len = 0
        for (ex_index, example) in enumerate(examples):
            tokens_b = None
            if forSequenceTagging:
                tokens_a, labels = tokenizer.tokenize_with_label_extension(example.text_a, example.labels,
                                                                           copy_previous_label=True)

                # FIND MAX LEN
                if len(tokens_a) > max_len:
                    max_len = len(tokens_a)

                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]
                    labels = labels[:(max_seq_length - 2)]
                labels = ["X"] + labels + ["X"]
            else:

                tokens_a = tokenizer.tokenize(example.text_a)

                # FIND MAX LEN
                if len(tokens_a) > max_len:
                    max_len = len(tokens_a)

                if example.text_b:
                    tokens_b = tokenizer.tokenize(example.text_b)
                    # Modifies `tokens_a` and `tokens_b` in place so that the total
                    # length is less than the specified length.
                    # Account for [CLS], [SEP], [SEP] with "- 3"
                    InputExample.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                else:
                    # Account for [CLS] and [SEP] with "- 2"
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = [cls_token] + tokens_a + [sep_token]

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if min_seq_length is not None:
                if len(input_ids) < min_seq_length:
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            if forSequenceTagging:
                label_ids = self.convert_labels_to_ids(labels)
                label_ids += padding
                assert len(label_ids) == max_seq_length
            else:
                label_list = self.get_labels()
                label_map = {label: i for i, label in enumerate(label_list)}
                label_ids = label_map[example.labels]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if forSequenceTagging:
                    logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
                else:
                    logger.info("label_id: %s" % " ".join(str(label_ids)))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))

        logger.info("MAX_LEN = {}".format(max_len + 2))
        return features

    @classmethod
    def features_to_dataset(cls, feature_list, isMultiChoice=None):

        if isMultiChoice:
            all_input_ids = torch.tensor(old_data_processors.MultiChoiceInputFeatures.select_field(feature_list, 'input_ids'),
                                         dtype=torch.long)
            all_input_mask = torch.tensor(old_data_processors.MultiChoiceInputFeatures.select_field(feature_list, 'input_mask'),
                                          dtype=torch.long)
            all_segment_ids = torch.tensor(old_data_processors.MultiChoiceInputFeatures.select_field(feature_list, 'segment_ids'),
                                           dtype=torch.long)
            all_label = torch.tensor([f.label for f in feature_list], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        else:
            all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_ids for f in feature_list], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_conll(cls, input_file, token_column=1, label_column=4, replace=None):
        """Reads a conll type file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            lines.append("\n")  # workaround adding a stop criteria for last sentence iteration

            sentences = []
            try:
                lines[0].split('\t')[label_column]
            except IndexError as err:
                print('Label column', err)
                raise

            tokenizer = BasicTokenizer()
            sent_tokens = []
            sent_labels = []

            for line in lines:

                line = line.split('\t')

                if len(line) < 2:
                    assert len(sent_tokens) == len(sent_labels)
                    if sent_tokens == []:
                        continue

                    if replace == None:
                        sentences.append([' '.join(sent_tokens), sent_labels])
                    else:
                        sent_labels = [replace[label] if label in replace.keys() else label for label in sent_labels]
                        sentences.append([' '.join(sent_tokens), sent_labels])
                    sent_tokens = []
                    sent_labels = []
                    continue

                token = line[token_column]
                label = line[label_column].replace('\n', '')
                tokenized = tokenizer.tokenize(token)

                if len(tokenized) > 1:

                    for i in range(len(tokenized)):
                        if 'B-' in label:
                            if i < 1:
                                sent_tokens.append(tokenized[i])
                                sent_labels.append(label)
                            else:
                                sent_tokens.append(tokenized[i])
                                # sent_labels.append(label.replace('B-', 'I-')) #if only the first token should be B-
                                sent_labels.append(label)
                        else:
                            sent_tokens.append(tokenized[i])
                            sent_labels.append(label)

                else:
                    sent_tokens.append(tokenized[0])
                    sent_labels.append(label)

        return sentences


class CDCPDataProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_docs(
            self.read_cdcp_docs(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_from_docs(
            self.read_cdcp_docs(os.path.join(data_dir, "dev")), "dev")

    def get_test_examples(self, data_dir, setname="test"):
        """See base class."""
        return self._create_examples_from_docs(
            self.read_cdcp_docs(os.path.join(data_dir, setname)), "test")

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__reason", "__label__evidence"]

    @staticmethod
    def read_cdcp_docs(data_dir):
        doc_ids = [file.split('.')[0] for file in os.listdir(data_dir) if file.split(".")[-1] == "txt"]
        return [CDCPArgumentationDoc(data_dir + '/' + doc_id) for doc_id in doc_ids]

    def _create_examples_from_docs(self, docs, set_type):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class ArgMinSeqTagProcessorCDCP(DataProcessor):
    """Processor for RCT data set (CDCP format)"""

    def __init__(self):
        self.labels = ["X", "B-Policy", "I-Policy", 'B-Value', 'I-Value', 'B-Testimony', 'I-Testimony', 'B-Fact',
                       'I-Fact', 'B-Reference', 'I-Reference', 'O']
        self.label_map = self._create_label_map()

    def _create_label_map(self):
        label_map = collections.OrderedDict()
        for i, label in enumerate(self.labels):
            label_map[label] = i
        return label_map

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "train.conll")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "dev.conll")), "dev")

    def get_test_examples(self, data_dir, setname="test.conll"):
        """See base class."""
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, setname)), "test")

    def get_labels(self):
        """ See base class."""
        return self.labels

    def convert_labels_to_ids(self, labels):
        idx_list = []
        for label in labels:
            idx_list.append(self.label_map[label])
        return idx_list

    def convert_ids_to_labels(self, idx_list):
        labels_list = []
        for idx in idx_list:
            labels_list.append([key for key in self.label_map.keys() if self.label_map[key] == idx][0])
        return labels_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, str(i))
            text_a = line[0]
            labels = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, labels=labels))
        return examples


class ArgMinRelClassProcessorCDCP(CDCPDataProcessor):

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__reason", "__label__evidence"]

    def _create_examples_from_docs(self, docs, set_type):
        counts = {'reason': 0,
                  'evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    else:
                        rel_type = 'noRel'
                    counts[rel_type] += 1

                    label = '__label__' + rel_type
                    guid = "%s-%s" % (set_type, label)

                    text_a = doc.raw_text[c_src[0]:c_src[1]]
                    text_b = doc.raw_text[c_trg[0]:c_trg[1]]

                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=label))
        logger.info(
            "Relation counts ({} set): Reason {} | Evidence {} | None/Inv {}".format(set_type, counts['reason'],
                                                                counts['evidence'], counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                        counts['policy'], counts['fact'], counts['testimony'], counts['value'], counts['reference']))
        return examples


class ArgMinRelClassProcessorCDCPJointLearningInverseLabels(CDCPDataProcessor):

    class InputFeaturesCDCPJointLearning(InputFeatures):

        def __init__(self, input_ids, input_mask, segment_ids, label_ids, relation_label, link_label,
                     source_label, target_label):
            super().__init__(input_ids, input_mask, segment_ids, label_ids)

            self.relation_label = relation_label
            self.link_label = link_label
            self.source_label = source_label
            self.target_label = target_label

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__reason", "__label__inv_reason", "__label__evidence",
                "__label__inv_evidence"]

    @staticmethod
    def get_component_labels():
        return ['policy', 'fact', 'testimony', 'value', 'reference']

    def _create_examples_from_docs(self, docs, set_type):

        counts = {'reason': 0,
                  'inv_reason': 0,
                  'evidence': 0,
                  'inv_evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    elif (c_trg_idx, c_src_idx) in doc.links_lists['locate']:
                        rel_type = "inv_" + doc.links_lists['link'][
                            doc.links_lists['locate'].index((c_trg_idx, c_src_idx))]
                    else:
                        rel_type = 'noRel'

                    counts[rel_type] += 1

                    # c_types = [doc.prop_labels[c_i_idx], doc.prop_labels[c_j_idx]]
                    relation = '__label__' + rel_type
                    source = doc.prop_labels[c_src_idx]
                    target = doc.prop_labels[c_trg_idx]
                    link = 1 if relation != '__label__noRel' else 0

                    label_list = self.get_labels()
                    label_map = {label: i for i, label in enumerate(label_list)}

                    labels = [label_map[relation], link, self.get_component_labels().index(source),
                              self.get_component_labels().index(target)]

                    guid = "%s-%s" % (set_type, relation)

                    text_a = doc.raw_text[c_src[0]:c_src[1]]
                    text_b = doc.raw_text[c_trg[0]:c_trg[1]]

                    examples.append(
                        InputExample(guid=guid,
                                     text_a=text_a,
                                     text_b=text_b,
                                     labels=labels))
        logger.info(
            "Relation counts ({} set): Reason {} | Inv_Reason {} | Evidence {} | Inv_Evidence {} | None {}".format(
                set_type, counts['reason'], counts['inv_reason'],
                counts['evidence'],
                counts['inv_evidence'], counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                                                                                                          counts[
                                                                                                              'policy'],
                                                                                                          counts[
                                                                                                              'fact'],
                                                                                                          counts[
                                                                                                              'testimony'],
                                                                                                          counts[
                                                                                                              'value'],
                                                                                                          counts[
                                                                                                              'reference']))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     logger=None,
                                     forSequenceTagging=False,
                                     min_seq_length=None,
                                     cls_token='[CLS]',
                                     sep_token_extra=False,
                                     sep_token='[SEP]'):

        """Loads a data file into a list of `InputBatch`s."""

        features = []
        max_len = 0
        for (ex_index, example) in enumerate(examples):

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)

            # Record max sequence len
            if len(tokens_a) > max_len:
                max_len = len(tokens_a)
            elif len(tokens_b) > max_len:
                max_len = len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            InputExample.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            tokens = [cls_token] + tokens_a + [sep_token]

            if isinstance(tokenizer, transformers.RobertaTokenizer):
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [0] * len(tokens)

            tokens += tokens_b + [sep_token]

            segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if min_seq_length is not None:
                if len(input_ids) < min_seq_length:
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            relation_label = example.labels[0]
            link_label = example.labels[1]
            source_label = example.labels[2]
            target_label = example.labels[3]


            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                logger.info("relation_label: %s" % " ".join(str(relation_label)))
                logger.info("link_label: %s" % " ".join(str(link_label)))
                logger.info("source_label: %s" % " ".join(str(source_label)))
                logger.info("target_label: %s" % " ".join(str(target_label)))

            features.append(
                self.InputFeaturesCDCPJointLearning(input_ids=input_ids,
                                                    input_mask=input_mask,
                                                    segment_ids=segment_ids,
                                                    label_ids=None,
                                                    relation_label=relation_label,
                                                    link_label=link_label,
                                                    source_label=source_label,
                                                    target_label=target_label))

        logger.info("Maximum sequence length = {}".format(max_len + 2))
        return features

    def features_to_dataset(self, feature_list, isMultiChoice=None):

        all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
        relation_labels = torch.tensor([f.relation_label for f in feature_list], dtype=torch.long)
        link_labels = torch.tensor([f.link_label for f in feature_list], dtype=torch.long)
        source_labels = torch.tensor([f.source_label for f in feature_list], dtype=torch.long)
        target_labels = torch.tensor([f.target_label for f in feature_list], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, relation_labels,
                                link_labels, source_labels, target_labels)

        return dataset


class ArgMinRelClassProcessorCDCPWithContext(CDCPDataProcessor):
    """Processor for the CDCP data set with context (for training)."""

    TAGS = {'src': [' [AC1] ', ' [/AC1] '],
            'trg': [' [AC2] ', ' [/AC2] ']}

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__reason", "__label__evidence"]

    def _create_examples_from_docs(self, docs, set_type):
        """Creates examples for the training and dev sets."""

        longest_example = 0
        counts = {'reason': 0,
                  'evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    else:
                        rel_type = 'noRel'
                    counts[rel_type] += 1

                    c_types = [doc.prop_labels[c_src_idx], doc.prop_labels[c_trg_idx]]
                    label = '__label__' + rel_type
                    guid = "%s-%s" % (set_type, label)
                    text_a = doc.raw_text
                    components = {'src': c_src,
                                  'trg': c_trg}
                    examples.append(
                        CDCPExample(guid=guid, text_a=text_a, components=components, labels=label))

                    if len(text_a) > longest_example:
                        longest_example = len(text_a)

        logging.info("Longest {} set example = {} chars".format(set_type, longest_example))
        logger.info(
            "Relation counts ({} set): Reason {} | Evidence {} | None/Inv {}".format(set_type, counts['reason'],
                                                                                     counts['evidence'],
                                                                                     counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                                                                                                          counts[
                                                                                                              'policy'],
                                                                                                          counts[
                                                                                                              'fact'],
                                                                                                          counts[
                                                                                                              'testimony'],
                                                                                                          counts[
                                                                                                              'value'],
                                                                                                          counts[
                                                                                                              'reference']))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     logger=None,
                                     forSequenceTagging=False,
                                     min_seq_length=None,
                                     cls_token='[CLS]',
                                     sep_token_extra=False,
                                     sep_token='[SEP]'):

        """Loads a data file into a list of `InputBatch`s."""

        features = []
        max_output_id_len = 0
        max_output = None
        total_time_collecting_tokens = 0
        total_time_converting_to_ids = 0
        num_skipped = 0
        for (ex_index, example) in enumerate(examples):

            components = example.components
            tokens = []
            c_order = ['src', 'trg'] if components['src'][0] < components['trg'][0] else ['trg', 'src']

            start_time = time.time()
            # Tokens before first component
            tokens += tokenizer.tokenize(example.text_a[:components[c_order[0]][0]])

            # First component
            tokens += [self.TAGS[c_order[0]][0]]
            tokens += tokenizer.tokenize(example.text_a[components[c_order[0]][0]:components[c_order[0]][1]])
            tokens += [self.TAGS[c_order[0]][1]]

            # Tokens between components
            tokens += tokenizer.tokenize(example.text_a[components[c_order[0]][1]:components[c_order[1]][0]])

            # Second component
            tokens += [self.TAGS[c_order[1]][0]]
            tokens += tokenizer.tokenize(example.text_a[components[c_order[1]][0]:components[c_order[1]][1]])
            tokens += [self.TAGS[c_order[1]][1]]

            # Tokens after last component
            tokens += tokenizer.tokenize(example.text_a[components[c_order[1]][1]:])

            # Add CLS and SEP
            tokens = [cls_token] + tokens + [sep_token]

            if len(tokens) > max_seq_length:
                # if components[c_order[0]]
                num_skipped += 1
                continue
                # tokens = tokens[:max_seq_length]

            total_time_collecting_tokens += (time.time() - start_time)

            start_time = time.time()
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            total_time_converting_to_ids += (time.time() - start_time)

            # DEBUG
            if len(input_ids) > max_output_id_len:
                max_output_id_len = len(input_ids)
                max_output = tokens

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids = [0] * len(input_ids)

            label_list = self.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}
            label_ids = label_map[example.labels]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label_id: %s" % " ".join(str(label_ids)))

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))

        logger.info("Max token length of sequence = {}".format(max_output_id_len))
        logger.info("Examples skipped = {}".format(num_skipped))
        logger.info("Time collecting tokens = {}".format(total_time_collecting_tokens))
        logger.info("Time converting to ids = {}".format(total_time_converting_to_ids))
        return features


class ArgMinRelClassProcessorCDCPRBERT(CDCPDataProcessor):
    """Processor for the CDCP data set with context (for training)."""

    TAGS = {'src': [' <src> ', ' </src> '],
            'trg': [' <trg> ', ' </trg> ']}

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__reason", "__label__evidence"]

    # TODO: Add component types to tags
    def create_text_with_tags(self, text, components):
        additional_len_from_tags = 0
        for component in components:
            for i in [0, 1]:
                before_tag = text[:component[1][i] + additional_len_from_tags]
                tag = self.TAGS[component[0]][i]
                after_tag = text[component[1][i] + additional_len_from_tags:]
                text = before_tag + tag + after_tag
                additional_len_from_tags += len(tag)
        return text

    def _create_examples_from_docs(self, docs, set_type):
        """Creates examples for the training and dev sets."""

        longest_example = 0
        counts = {'reason': 0,
                  'evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    else:
                        rel_type = 'noRel'
                    counts[rel_type] += 1

                    c_text_order = [['src', c_src], ['trg', c_trg]] if c_src_idx < c_trg_idx else [['trg', c_trg], ['src', c_src]]
                    label = self.get_labels().index('__label__' + rel_type)
                    guid = "%s-%s" % (set_type, label)
                    text_a = self.create_text_with_tags(doc.raw_text, c_text_order)

                    examples.append(
                        InputExample(guid=guid, text_a=text_a, labels=label))

                    if len(text_a) > longest_example:
                        longest_example = len(text_a)

        logging.info("Longest {} set example = {} chars".format(set_type, longest_example))
        logger.info(
            "Relation counts ({} set): Reason {} | Evidence {} | None/Inv {}".format(set_type, counts['reason'],
                                                                counts['evidence'], counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                        counts['policy'], counts['fact'], counts['testimony'], counts['value'], counts['reference']))
        return examples

    # logger = logger, forSequenceTagging = forSequenceTagging, min_seq_length = 5
    def convert_examples_to_features(
            self,
            examples,
            max_seq_length,
            tokenizer,
            logger=None,
            forSequenceTagging=False,
            min_seq_length=5,
            cls_token="[CLS]",
            cls_token_segment_id=0,
            sep_token="[SEP]",
            pad_token=0,
            pad_token_segment_id=0,
            sequence_a_segment_id=0,
            add_sep_token=False,
            mask_padding_with_zero=True,
    ):
        current_max_seq_len = 0
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.tokenize(example.text_a)

            e11_p = tokens_a.index(self.TAGS['src'][0].strip())  # the start position of src entity
            e12_p = tokens_a.index(self.TAGS['src'][1].strip())  # the end position of src entity
            e21_p = tokens_a.index(self.TAGS['trg'][0].strip())  # the start position of trg entity
            e22_p = tokens_a.index(self.TAGS['trg'][1].strip())  # the end position of trg entity

            # Replace the token
            tokens_a[e11_p] = "$"
            tokens_a[e12_p] = "$"
            tokens_a[e21_p] = "#"
            tokens_a[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            if add_sep_token:
                special_tokens_count = 2
            else:
                special_tokens_count = 1

            tokens = tokens_a
            if add_sep_token:
                tokens += [sep_token]

            token_type_ids = [sequence_a_segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            if len(tokens) > max_seq_length:
                continue
            elif len(tokens) > current_max_seq_len:
                current_max_seq_len = len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
            assert len(attention_mask) == max_seq_length, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_length
            )
            assert len(token_type_ids) == max_seq_length, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_length
            )

            label_id = int(example.labels)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s (id = %d)" % (example.labels, label_id))
                logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
                logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

            features.append(
                RBERTInputFeatures(
                    example_id=ex_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label_id=label_id,
                    e1_mask=e1_mask,
                    e2_mask=e2_mask,
                )
            )

        logger.info("Longest total sequence: {}".format(current_max_seq_len))
        return features

    def features_to_dataset(self, features, isMultiChoice=None):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask

        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_label_ids,
            all_e1_mask,
            all_e2_mask,
        )
        return dataset


class ArgMinRelClassProcessorCDCPRBERTJointLearning(CDCPDataProcessor):
    """Processor for the CDCP data set with context (for training)."""

    class RBERTInputFeaturesJointLearning(RBERTInputFeatures):
        def __init__(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, relation_label, source_label,
                     target_label, label_id=None):
            super().__init__(input_ids, attention_mask, token_type_ids, label_id, e1_mask, e2_mask)

            self.relation_label = relation_label
            self.source_label = source_label
            self.target_label = target_label

    TAGS = {'src': [' <src> ', ' </src> '],
            'trg': [' <trg> ', ' </trg> ']}

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__reason", "__label__evidence"]

    @staticmethod
    def get_component_labels():
        return ['policy', 'fact', 'testimony', 'value', 'reference']

    # TODO: Add component types to tags
    def create_text_with_tags(self, text, components):
        additional_len_from_tags = 0
        for component in components:
            for i in [0, 1]:
                before_tag = text[:component[1][i] + additional_len_from_tags]
                tag = self.TAGS[component[0]][i]
                after_tag = text[component[1][i] + additional_len_from_tags:]
                text = before_tag + tag + after_tag
                additional_len_from_tags += len(tag)
        return text

    def _create_examples_from_docs(self, docs, set_type):
        """Creates examples for the training and dev sets."""

        longest_example = 0
        counts = {'reason': 0,
                  'evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    else:
                        rel_type = 'noRel'
                    counts[rel_type] += 1

                    relation = self.get_labels().index('__label__' + rel_type)
                    source = doc.prop_labels[c_src_idx]
                    target = doc.prop_labels[c_trg_idx]
                    label_list = self.get_labels()
                    labels = [relation, self.get_component_labels().index(source), self.get_component_labels().index(target)]

                    c_text_order = [['src', c_src], ['trg', c_trg]] if c_src_idx < c_trg_idx else [['trg', c_trg], ['src', c_src]]
                    guid = "%s-%s" % (set_type, relation)
                    text_a = self.create_text_with_tags(doc.raw_text, c_text_order)

                    examples.append(
                        InputExample(guid=guid, text_a=text_a, labels=labels))

                    if len(text_a) > longest_example:
                        longest_example = len(text_a)

        logging.info("Longest {} set example = {} chars".format(set_type, longest_example))
        logger.info(
            "Relation counts ({} set): Reason {} | Evidence {} | None/Inv {}".format(set_type, counts['reason'],
                                                                counts['evidence'], counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                        counts['policy'], counts['fact'], counts['testimony'], counts['value'], counts['reference']))
        return examples

    # logger = logger, forSequenceTagging = forSequenceTagging, min_seq_length = 5
    def convert_examples_to_features(
            self,
            examples,
            max_seq_length,
            tokenizer,
            logger=None,
            forSequenceTagging=False,
            min_seq_length=5,
            cls_token="[CLS]",
            cls_token_segment_id=0,
            sep_token="[SEP]",
            pad_token=0,
            pad_token_segment_id=0,
            sequence_a_segment_id=0,
            add_sep_token=False,
            mask_padding_with_zero=True,
    ):
        current_max_seq_len = 0
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.tokenize(example.text_a)

            e11_p = tokens_a.index(self.TAGS['src'][0].strip())  # the start position of src entity
            e12_p = tokens_a.index(self.TAGS['src'][1].strip())  # the end position of src entity
            e21_p = tokens_a.index(self.TAGS['trg'][0].strip())  # the start position of trg entity
            e22_p = tokens_a.index(self.TAGS['trg'][1].strip())  # the end position of trg entity

            # Replace the token
            tokens_a[e11_p] = "$"
            tokens_a[e12_p] = "$"
            tokens_a[e21_p] = "#"
            tokens_a[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            if add_sep_token:
                special_tokens_count = 2
            else:
                special_tokens_count = 1

            tokens = tokens_a
            if add_sep_token:
                tokens += [sep_token]

            token_type_ids = [sequence_a_segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids

            if len(tokens) > max_seq_length:
                continue
            elif len(tokens) > current_max_seq_len:
                current_max_seq_len = len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            relation_label = example.labels[0]
            source_label = example.labels[1]
            target_label = example.labels[2]

            assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
            assert len(attention_mask) == max_seq_length, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_length
            )
            assert len(token_type_ids) == max_seq_length, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_length
            )

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("relation_label: %s (id = %d)" % (example.labels, relation_label))
                logger.info("source_label: %s (id = %d)" % (example.labels, source_label))
                logger.info("target_label: %s (id = %d)" % (example.labels, target_label))
                logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
                logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

            features.append(
                self.RBERTInputFeaturesJointLearning(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    e1_mask=e1_mask,
                    e2_mask=e2_mask,
                    relation_label=relation_label,
                    source_label=source_label,
                    target_label=target_label
                )
            )

        logger.info("Longest total sequence: {}".format(current_max_seq_len))
        return features

    def features_to_dataset(self, features, isMultiChoice=None):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask

        relation_labels = torch.tensor([f.relation_label for f in features], dtype=torch.long)
        source_labels = torch.tensor([f.source_label for f in features], dtype=torch.long)
        target_labels = torch.tensor([f.target_label for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_e1_mask,
            all_e2_mask,
            relation_labels,
            source_labels,
            target_labels
        )
        return dataset


class ArgMinRelClassProcessorCDCPWithDistance(CDCPDataProcessor):

    class ExampleWithDistance(InputExample):

        def __init__(self, guid, text_a, text_b, distance, labels=None):
            super().__init__(guid, text_a, text_b, labels=labels)

            self.distance = distance

    class InputFeaturesWithDistance(InputFeatures):

        def __init__(self, input_ids, input_mask, segment_ids, label_ids, distance_encoding):
            super().__init__(input_ids, input_mask, segment_ids, label_ids)

            self.distance_encoding = distance_encoding

    def _create_examples_from_docs(self, docs, set_type):

        counts = {'reason': 0,
                  'evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    else:
                        rel_type = 'noRel'
                    counts[rel_type] += 1

                    # c_types = [doc.prop_labels[c_i_idx], doc.prop_labels[c_j_idx]]
                    label = '__label__' + rel_type
                    guid = "%s-%s" % (set_type, label)

                    text_a = doc.raw_text[c_src[0]:c_src[1]]
                    text_b = doc.raw_text[c_trg[0]:c_trg[1]]

                    # Distance between components in document up to maximum of 5
                    distance = c_trg_idx - c_src_idx
                    distance = 5 if distance > 5 else distance
                    distance = -5 if distance < -5 else distance

                    examples.append(
                        self.ExampleWithDistance(guid=guid,
                                                 text_a=text_a,
                                                 text_b=text_b,
                                                 distance=distance,
                                                 labels=label))
        logger.info(
            "Relation counts ({} set): Reason {} | Evidence {} | None/Inv {}".format(set_type, counts['reason'],
                                                                counts['evidence'], counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                        counts['policy'], counts['fact'], counts['testimony'], counts['value'], counts['reference']))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     logger=None,
                                     forSequenceTagging=False,
                                     min_seq_length=None,
                                     cls_token='[CLS]',
                                     sep_token_extra=False,
                                     sep_token='[SEP]'):

        """Loads a data file into a list of `InputBatch`s."""

        features = []
        max_len = 0
        for (ex_index, example) in enumerate(examples):

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)

            # Record max sequence len
            if len(tokens_a) > max_len:
                max_len = len(tokens_a)
            elif len(tokens_b) > max_len:
                max_len = len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            InputExample.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            tokens = [cls_token] + tokens_a + [sep_token]

            if isinstance(tokenizer, transformers.RobertaTokenizer):
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [0] * len(tokens)

            tokens += tokens_b + [sep_token]

            segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if min_seq_length is not None:
                if len(input_ids) < min_seq_length:
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            label_list = self.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}
            label_ids = label_map[example.labels]

            # From Galassi et al https://aclanthology.org/W18-5201.pdf
            # We represent distance using as a 10-bit array, where the first 5 bits are
            # used in case that the source precedes the target,
            # and the last 5 bits are used in the opposite case.
            # In both cases, the number of consecutive “1” values encodes the value of the distance (distances
            # are capped by 5). For example, if the target precedes the source by two sentences, the distance is
            # −2, which produces encoding 0001100000; if the
            # source precedes the target by three sentences, the
            # distance is 3, with encoding 0000011100.

            distance_encoding = ([0, 0, 0, 0, 0]
                                 + [1 for _ in range(abs(example.distance))]
                                 + [0 for _ in range(5 - abs(example.distance))]) if example.distance > 0 else (
                                [0 for _ in range(5 - abs(example.distance))]
                                + [1 for _ in range(abs(example.distance))]
                                + [0, 0, 0, 0, 0])

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(distance_encoding) == 10

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info(
                    "distance_encoding: %s" % " ".join([str(x) for x in distance_encoding]))
                logger.info("label_id: %s" % " ".join(str(label_ids)))

            features.append(
                self.InputFeaturesWithDistance(input_ids=input_ids,
                                               input_mask=input_mask,
                                               segment_ids=segment_ids,
                                               label_ids=label_ids,
                                               distance_encoding=distance_encoding))

        logger.info("Maximum sequence length = {}".format(max_len + 2))
        return features

    def features_to_dataset(self, feature_list, isMultiChoice=None):

        all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in feature_list], dtype=torch.long)
        distance_encoding = torch.tensor([f.distance_encoding for f in feature_list], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, distance_encoding)

        return dataset

class ArgMinRelClassProcessorCDCPWithDistanceAndComponentType(CDCPDataProcessor):

    COMPONENT_TYPE_ENCODINGS = {"policy": [1, 0, 0, 0],
                                "value": [0, 1, 0, 0],
                                "fact": [0, 0, 1, 0],
                                "testimony": [0, 0, 0, 1],
                                "reference": [0, 0, 0, 0]}

    class ExampleWithDistanceAndComponentType(InputExample):

        def __init__(self, guid, text_a, text_b, distance, component_types, labels=None):
            super().__init__(guid, text_a, text_b, labels=labels)

            self.distance = distance
            self.component_types = component_types

    class InputFeaturesWithDistanceAndComponentType(InputFeatures):

        def __init__(self, input_ids, input_mask, segment_ids, label_ids, distance_encoding, component_encoding):
            super().__init__(input_ids, input_mask, segment_ids, label_ids)

            self.distance_encoding = distance_encoding
            self.component_encoding = component_encoding

    def _create_examples_from_docs(self, docs, set_type):
        counts = {'reason': 0,
                  'evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    else:
                        rel_type = 'noRel'
                    counts[rel_type] += 1

                    label = '__label__' + rel_type
                    guid = "%s-%s" % (set_type, label)

                    text_a = doc.raw_text[c_src[0]:c_src[1]]
                    text_b = doc.raw_text[c_trg[0]:c_trg[1]]

                    # Distance between components in document up to maximum of 5
                    distance = c_trg_idx - c_src_idx
                    distance = 5 if distance > 5 else distance
                    distance = -5 if distance < -5 else distance

                    component_types = [doc.prop_labels[c_src_idx], doc.prop_labels[c_trg_idx]]

                    examples.append(
                        self.ExampleWithDistanceAndComponentType(guid=guid,
                                                                 text_a=text_a,
                                                                 text_b=text_b,
                                                                 distance=distance,
                                                                 component_types=component_types,
                                                                 labels=label))
        logger.info(
            "Relation counts ({} set): Reason {} | Evidence {} | None/Inv {}".format(set_type, counts['reason'],
                                                                counts['evidence'], counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                        counts['policy'], counts['fact'], counts['testimony'], counts['value'], counts['reference']))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     logger=None,
                                     forSequenceTagging=False,
                                     min_seq_length=None,
                                     cls_token='[CLS]',
                                     sep_token_extra=False,
                                     sep_token='[SEP]'):

        """Loads a data file into a list of `InputBatch`s."""

        features = []
        max_len = 0
        for (ex_index, example) in enumerate(examples):

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)

            # Record max sequence len
            if len(tokens_a) > max_len:
                max_len = len(tokens_a)
            elif len(tokens_b) > max_len:
                max_len = len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            InputExample.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            tokens = [cls_token] + tokens_a + [sep_token]

            if isinstance(tokenizer, transformers.RobertaTokenizer):
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [0] * len(tokens)

            tokens += tokens_b + [sep_token]

            segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if min_seq_length is not None:
                if len(input_ids) < min_seq_length:
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            label_list = self.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}
            label_ids = label_map[example.labels]

            # From Galassi et al https://aclanthology.org/W18-5201.pdf
            # We represent distance using as a 10-bit array, where the first 5 bits are
            # used in case that the source precedes the target,
            # and the last 5 bits are used in the opposite case.
            # In both cases, the number of consecutive “1” values encodes the value of the distance (distances
            # are capped by 5). For example, if the target precedes the source by two sentences, the distance is
            # −2, which produces encoding 0001100000; if the
            # source precedes the target by three sentences, the
            # distance is 3, with encoding 0000011100.

            distance_encoding = ([0, 0, 0, 0, 0]
                                 + [1 for _ in range(abs(example.distance))]
                                 + [0 for _ in range(5 - abs(example.distance))]) if example.distance > 0 else (
                                [0 for _ in range(5 - abs(example.distance))]
                                + [1 for _ in range(abs(example.distance))]
                                + [0, 0, 0, 0, 0])

            component_encoding = self.COMPONENT_TYPE_ENCODINGS[example.component_types[0]] + \
                                 self.COMPONENT_TYPE_ENCODINGS[example.component_types[1]]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(distance_encoding) == 10
            assert len(component_encoding) == 8

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info(
                    "distance_encoding: %s" % " ".join([str(x) for x in distance_encoding]))
                logger.info("label_id: %s" % " ".join(str(label_ids)))

            features.append(
                self.InputFeaturesWithDistanceAndComponentType(input_ids=input_ids,
                                                               input_mask=input_mask,
                                                               segment_ids=segment_ids,
                                                               label_ids=label_ids,
                                                               distance_encoding=distance_encoding,
                                                               component_encoding=component_encoding))

        logger.info("Maximum sequence length = {}".format(max_len + 2))
        return features

    def features_to_dataset(self, feature_list, isMultiChoice=None):

        all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in feature_list], dtype=torch.long)
        distance_encoding = torch.tensor([f.distance_encoding for f in feature_list], dtype=torch.long)
        component_encoding = torch.tensor([f.component_encoding for f in feature_list], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, distance_encoding,
                                component_encoding)

        return dataset


class ArgMinRelClassProcessorCDCPResNet(CDCPDataProcessor):

    class ExampleWithDistance(InputExample):

        def __init__(self, guid, text_a, text_b, distance, labels=None):
            super().__init__(guid, text_a, text_b, labels=labels)

            self.distance = distance

    class InputFeaturesWithDistance(InputFeatures):

        def __init__(self, input_ids, input_mask, segment_ids, label_ids, distance_encoding):
            super().__init__(input_ids, input_mask, segment_ids, label_ids)

            self.distance_encoding = distance_encoding

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__reason", "__label__evidence"]

    def _create_examples_from_docs(self, docs, set_type):
        counts = {'reason': 0,
                  'evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    else:
                        rel_type = 'noRel'
                    counts[rel_type] += 1

                    # c_types = [doc.prop_labels[c_i_idx], doc.prop_labels[c_j_idx]]
                    # if c_src_idx < c_trg_idx or rel_type == 'noRel' else '__label__inv_' + rel_type
                    label = '__label__' + rel_type
                    guid = "%s-%s" % (set_type, label)

                    text_a = doc.raw_text[c_src[0]:c_src[1]]
                    text_b = doc.raw_text[c_trg[0]:c_trg[1]]

                    # Distance between components in document up to maximum of 5
                    distance = c_trg_idx - c_src_idx
                    distance = 5 if distance > 5 else distance
                    distance = -5 if distance < -5 else distance

                    examples.append(
                        self.ExampleWithDistance(guid=guid,
                                                 text_a=text_a,
                                                 text_b=text_b,
                                                 distance=distance,
                                                 labels=label))
        logger.info(
            "Relation counts ({} set): Reason {} | Evidence {} | None/Inv {}".format(set_type, counts['reason'],
                                                                counts['evidence'], counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                        counts['policy'], counts['fact'], counts['testimony'], counts['value'], counts['reference']))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     logger=None,
                                     forSequenceTagging=False,
                                     min_seq_length=None,
                                     cls_token='[CLS]',
                                     sep_token_extra=False,
                                     sep_token='[SEP]'):

        """Loads a data file into a list of `InputBatch`s."""

        features = []
        max_len = 0
        for (ex_index, example) in enumerate(examples):

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)

            # Record max sequence len
            if len(tokens_a) > max_len:
                max_len = len(tokens_a)
            elif len(tokens_b) > max_len:
                max_len = len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            InputExample.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            tokens = [cls_token] + tokens_a + [sep_token]

            if isinstance(tokenizer, transformers.RobertaTokenizer):
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [0] * len(tokens)

            tokens += tokens_b + [sep_token]

            segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if min_seq_length is not None:
                if len(input_ids) < min_seq_length:
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            label_list = self.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}
            label_ids = label_map[example.labels]

            # From Galassi et al https://aclanthology.org/W18-5201.pdf
            # We represent distance using as a 10-bit array, where the first 5 bits are
            # used in case that the source precedes the target,
            # and the last 5 bits are used in the opposite case.
            # In both cases, the number of consecutive “1” values encodes the value of the distance (distances
            # are capped by 5). For example, if the target precedes the source by two sentences, the distance is
            # −2, which produces encoding 0001100000; if the
            # source precedes the target by three sentences, the
            # distance is 3, with encoding 0000011100.

            distance_encoding = ([0, 0, 0, 0, 0]
                                 + [1 for _ in range(abs(example.distance))]
                                 + [0 for _ in range(5 - abs(example.distance))]) if example.distance > 0 else (
                                [0 for _ in range(5 - abs(example.distance))]
                                + [1 for _ in range(abs(example.distance))]
                                + [0, 0, 0, 0, 0])

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(distance_encoding) == 10

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info(
                    "distance_encoding: %s" % " ".join([str(x) for x in distance_encoding]))
                logger.info("label_id: %s" % " ".join(str(label_ids)))

            features.append(
                self.InputFeaturesWithDistance(input_ids=input_ids,
                                               input_mask=input_mask,
                                               segment_ids=segment_ids,
                                               label_ids=label_ids,
                                               distance_encoding=distance_encoding))

        logger.info("Maximum sequence length = {}".format(max_len + 2))
        return features

    def features_to_dataset(self, feature_list, isMultiChoice=None):

        all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in feature_list], dtype=torch.long)
        distance_encoding = torch.tensor([f.distance_encoding for f in feature_list], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, distance_encoding)

        return dataset


class ArgMinRelClassProcessorCDCPResNetJointLearning(CDCPDataProcessor):

    class ExampleResNetJointLearning(InputExample):

        def __init__(self, guid, text_a, text_b, distance, labels=None):
            super().__init__(guid, text_a, text_b, labels=labels)

            self.distance = distance

    class InputFeaturesResNetJointLearning(InputFeatures):

        def __init__(self, input_ids, input_mask, segment_ids, label_ids, distance_encoding, relation_label, link_label,
                     source_label, target_label):
            super().__init__(input_ids, input_mask, segment_ids, label_ids)

            self.distance_encoding = distance_encoding
            self.relation_label = relation_label
            self.link_label = link_label
            self.source_label = source_label
            self.target_label = target_label

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__reason", "__label__evidence"]

    @staticmethod
    def get_component_labels():
        return ['policy', 'fact', 'testimony', 'value', 'reference']

    def _create_examples_from_docs(self, docs, set_type):

        counts = {'reason': 0,
                  'evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    else:
                        rel_type = 'noRel'
                    counts[rel_type] += 1

                    # c_types = [doc.prop_labels[c_i_idx], doc.prop_labels[c_j_idx]]
                    relation = '__label__' + rel_type
                    source = doc.prop_labels[c_src_idx]
                    target = doc.prop_labels[c_trg_idx]
                    link = 1 if relation != '__label__noRel' else 0

                    label_list = self.get_labels()
                    label_map = {label: i for i, label in enumerate(label_list)}

                    labels = [label_map[relation], link, self.get_component_labels().index(source), self.get_component_labels().index(target)]

                    guid = "%s-%s" % (set_type, relation)

                    text_a = doc.raw_text[c_src[0]:c_src[1]]
                    text_b = doc.raw_text[c_trg[0]:c_trg[1]]

                    # Distance between components in document up to maximum of 5
                    distance = c_trg_idx - c_src_idx
                    distance = 5 if distance > 5 else distance
                    distance = -5 if distance < -5 else distance

                    examples.append(
                        self.ExampleResNetJointLearning(guid=guid,
                                                 text_a=text_a,
                                                 text_b=text_b,
                                                 distance=distance,
                                                 labels=labels))

        logger.info(
            "Relation counts ({} set): Reason {} | Evidence {} | None/Inv {}".format(set_type, counts['reason'],
                                                                counts['evidence'], counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                        counts['policy'], counts['fact'], counts['testimony'], counts['value'], counts['reference']))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     logger=None,
                                     forSequenceTagging=False,
                                     min_seq_length=None,
                                     cls_token='[CLS]',
                                     sep_token_extra=False,
                                     sep_token='[SEP]'):

        """Loads a data file into a list of `InputBatch`s."""

        features = []
        max_len = 0
        for (ex_index, example) in enumerate(examples):

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)

            # Record max sequence len
            if len(tokens_a) > max_len:
                max_len = len(tokens_a)
            elif len(tokens_b) > max_len:
                max_len = len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            InputExample.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            tokens = [cls_token] + tokens_a + [sep_token]

            if isinstance(tokenizer, transformers.RobertaTokenizer):
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [0] * len(tokens)

            tokens += tokens_b + [sep_token]

            segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if min_seq_length is not None:
                if len(input_ids) < min_seq_length:
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            relation_label = example.labels[0]
            link_label = example.labels[1]
            source_label = example.labels[2]
            target_label = example.labels[3]

            # From Galassi et al https://aclanthology.org/W18-5201.pdf
            # We represent distance using as a 10-bit array, where the first 5 bits are
            # used in case that the source precedes the target,
            # and the last 5 bits are used in the opposite case.
            # In both cases, the number of consecutive “1” values encodes the value of the distance (distances
            # are capped by 5). For example, if the target precedes the source by two sentences, the distance is
            # −2, which produces encoding 0001100000; if the
            # source precedes the target by three sentences, the
            # distance is 3, with encoding 0000011100.

            distance_encoding = ([0, 0, 0, 0, 0]
                                 + [1 for _ in range(abs(example.distance))]
                                 + [0 for _ in range(5 - abs(example.distance))]) if example.distance > 0 else (
                                [0 for _ in range(5 - abs(example.distance))]
                                + [1 for _ in range(abs(example.distance))]
                                + [0, 0, 0, 0, 0])

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(distance_encoding) == 10

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info(
                    "distance_encoding: %s" % " ".join([str(x) for x in distance_encoding]))
                logger.info("relation_label: %s" % " ".join(str(relation_label)))
                logger.info("link_label: %s" % " ".join(str(link_label)))
                logger.info("source_label: %s" % " ".join(str(source_label)))
                logger.info("target_label: %s" % " ".join(str(target_label)))

            features.append(
                self.InputFeaturesResNetJointLearning(input_ids=input_ids,
                                                      input_mask=input_mask,
                                                      segment_ids=segment_ids,
                                                      label_ids=None,
                                                      distance_encoding=distance_encoding,
                                                      relation_label=relation_label,
                                                      link_label=link_label,
                                                      source_label=source_label,
                                                      target_label=target_label))

        logger.info("Maximum sequence length = {}".format(max_len + 2))
        return features

    def features_to_dataset(self, feature_list, isMultiChoice=None):

        all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
        distance_encoding = torch.tensor([f.distance_encoding for f in feature_list], dtype=torch.long)
        relation_labels = torch.tensor([f.relation_label for f in feature_list], dtype=torch.long)
        link_labels = torch.tensor([f.link_label for f in feature_list], dtype=torch.long)
        source_labels = torch.tensor([f.source_label for f in feature_list], dtype=torch.long)
        target_labels = torch.tensor([f.target_label for f in feature_list], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, distance_encoding, relation_labels,
                                link_labels, source_labels, target_labels)

        return dataset


class ArgMinRelClassProcessorCDCPResNetJointLearningInverseLabels(CDCPDataProcessor):

    class ExampleResNetJointLearning(InputExample):

        def __init__(self, guid, text_a, text_b, distance, labels=None):
            super().__init__(guid, text_a, text_b, labels=labels)

            self.distance = distance

    class InputFeaturesResNetJointLearning(InputFeatures):

        def __init__(self, input_ids, input_mask, segment_ids, label_ids, distance_encoding, relation_label, link_label,
                     source_label, target_label):
            super().__init__(input_ids, input_mask, segment_ids, label_ids)

            self.distance_encoding = distance_encoding
            self.relation_label = relation_label
            self.link_label = link_label
            self.source_label = source_label
            self.target_label = target_label

    def get_labels(self):
        """See base class."""
        return ["__label__noRel", "__label__reason", "__label__inv_reason", "__label__evidence", "__label__inv_evidence"]

    @staticmethod
    def get_component_labels():
        return ['policy', 'fact', 'testimony', 'value', 'reference']

    def _create_examples_from_docs(self, docs, set_type):

        counts = {'reason': 0,
                  'inv_reason': 0,
                  'evidence': 0,
                  'inv_evidence': 0,
                  'noRel': 0,
                  'policy': 0, 'fact': 0, 'testimony': 0, 'value': 0, 'reference': 0}

        examples = []
        for doc in docs:
            for prop_idx, prop in enumerate(doc.prop_offsets):
                counts[doc.prop_labels[prop_idx]] += 1
            for c_src_idx, c_src in enumerate(doc.prop_offsets):
                for c_trg_idx, c_trg in enumerate(doc.prop_offsets):
                    if c_src_idx == c_trg_idx:
                        continue
                    if (c_src_idx, c_trg_idx) in doc.links_lists['locate']:
                        rel_type = doc.links_lists['link'][doc.links_lists['locate'].index((c_src_idx, c_trg_idx))]
                    elif (c_trg_idx, c_src_idx) in doc.links_lists['locate']:
                        rel_type = "inv_" + doc.links_lists['link'][doc.links_lists['locate'].index((c_trg_idx, c_src_idx))]
                    else:
                        rel_type = 'noRel'

                    counts[rel_type] += 1

                    # c_types = [doc.prop_labels[c_i_idx], doc.prop_labels[c_j_idx]]
                    relation = '__label__' + rel_type
                    source = doc.prop_labels[c_src_idx]
                    target = doc.prop_labels[c_trg_idx]
                    link = 1 if relation != '__label__noRel' else 0

                    label_list = self.get_labels()
                    label_map = {label: i for i, label in enumerate(label_list)}

                    labels = [label_map[relation], link, self.get_component_labels().index(source), self.get_component_labels().index(target)]

                    guid = "%s-%s" % (set_type, relation)

                    text_a = doc.raw_text[c_src[0]:c_src[1]]
                    text_b = doc.raw_text[c_trg[0]:c_trg[1]]

                    # Distance between components in document up to maximum of 5
                    distance = c_trg_idx - c_src_idx
                    distance = 5 if distance > 5 else distance
                    distance = -5 if distance < -5 else distance



                    examples.append(
                        self.ExampleResNetJointLearning(guid=guid,
                                                 text_a=text_a,
                                                 text_b=text_b,
                                                 distance=distance,
                                                 labels=labels))
        logger.info(
            "Relation counts ({} set): Reason {} | Inv_Reason {} | Evidence {} | Inv_Evidence {} | None {}".format(
                set_type, counts['reason'], counts['inv_reason'],
                counts['evidence'],
                counts['inv_evidence'], counts['noRel']))
        logger.info(
            "Entity counts ({} set): Policy {} | Fact {} | Testimony {} | Value {} | Reference {}".format(set_type,
                        counts['policy'], counts['fact'], counts['testimony'], counts['value'], counts['reference']))
        return examples

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer,
                                     logger=None,
                                     forSequenceTagging=False,
                                     min_seq_length=None,
                                     cls_token='[CLS]',
                                     sep_token_extra=False,
                                     sep_token='[SEP]'):

        """Loads a data file into a list of `InputBatch`s."""

        features = []
        max_len = 0
        for (ex_index, example) in enumerate(examples):

            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)

            # Record max sequence len
            if len(tokens_a) > max_len:
                max_len = len(tokens_a)
            elif len(tokens_b) > max_len:
                max_len = len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            InputExample.truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            tokens = [cls_token] + tokens_a + [sep_token]

            if isinstance(tokenizer, transformers.RobertaTokenizer):
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]

            segment_ids = [0] * len(tokens)

            tokens += tokens_b + [sep_token]

            segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if min_seq_length is not None:
                if len(input_ids) < min_seq_length:
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            relation_label = example.labels[0]
            link_label = example.labels[1]
            source_label = example.labels[2]
            target_label = example.labels[3]

            # From Galassi et al https://aclanthology.org/W18-5201.pdf
            # We represent distance using as a 10-bit array, where the first 5 bits are
            # used in case that the source precedes the target,
            # and the last 5 bits are used in the opposite case.
            # In both cases, the number of consecutive “1” values encodes the value of the distance (distances
            # are capped by 5). For example, if the target precedes the source by two sentences, the distance is
            # −2, which produces encoding 0001100000; if the
            # source precedes the target by three sentences, the
            # distance is 3, with encoding 0000011100.

            distance_encoding = ([0, 0, 0, 0, 0]
                                 + [1 for _ in range(abs(example.distance))]
                                 + [0 for _ in range(5 - abs(example.distance))]) if example.distance > 0 else (
                                [0 for _ in range(5 - abs(example.distance))]
                                + [1 for _ in range(abs(example.distance))]
                                + [0, 0, 0, 0, 0])

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(distance_encoding) == 10

            if ex_index < 10 and logger is not None:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info(
                    "distance_encoding: %s" % " ".join([str(x) for x in distance_encoding]))
                logger.info("relation_label: %s" % " ".join(str(relation_label)))
                logger.info("link_label: %s" % " ".join(str(link_label)))
                logger.info("source_label: %s" % " ".join(str(source_label)))
                logger.info("target_label: %s" % " ".join(str(target_label)))

            features.append(
                self.InputFeaturesResNetJointLearning(input_ids=input_ids,
                                                      input_mask=input_mask,
                                                      segment_ids=segment_ids,
                                                      label_ids=None,
                                                      distance_encoding=distance_encoding,
                                                      relation_label=relation_label,
                                                      link_label=link_label,
                                                      source_label=source_label,
                                                      target_label=target_label))

        logger.info("Maximum sequence length = {}".format(max_len + 2))
        return features

    def features_to_dataset(self, feature_list, isMultiChoice=None):

        all_input_ids = torch.tensor([f.input_ids for f in feature_list], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature_list], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature_list], dtype=torch.long)
        distance_encoding = torch.tensor([f.distance_encoding for f in feature_list], dtype=torch.long)
        relation_labels = torch.tensor([f.relation_label for f in feature_list], dtype=torch.long)
        link_labels = torch.tensor([f.link_label for f in feature_list], dtype=torch.long)
        source_labels = torch.tensor([f.source_label for f in feature_list], dtype=torch.long)
        target_labels = torch.tensor([f.target_label for f in feature_list], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, distance_encoding, relation_labels,
                                link_labels, source_labels, target_labels)

        return dataset


processors = {
    "cdcp_seqtag": ArgMinSeqTagProcessorCDCP,
    "cdcp_relclass": ArgMinRelClassProcessorCDCP,
    "cdcp_relclass_jl_inv": ArgMinRelClassProcessorCDCPJointLearningInverseLabels,
    "cdcp_relclass_context": ArgMinRelClassProcessorCDCPWithContext,
    "cdcp_relclass_rbert": ArgMinRelClassProcessorCDCPRBERT,
    "cdcp_relclass_rbert_jl": ArgMinRelClassProcessorCDCPRBERTJointLearning,
    "cdcp_relclass_distance": ArgMinRelClassProcessorCDCPWithDistance,
    "cdcp_relclass_distance_components": ArgMinRelClassProcessorCDCPWithDistanceAndComponentType,
    "cdcp_relclass_resnet": ArgMinRelClassProcessorCDCPResNet,
    "cdcp_relclass_resnet_jl": ArgMinRelClassProcessorCDCPResNetJointLearning,
    "cdcp_relclass_resnet_jl_inv": ArgMinRelClassProcessorCDCPResNetJointLearningInverseLabels
}

output_modes = {
    "cdcp_seqtag": SEQUENCE_TAGGING,
    "cdcp_relclass": CLASSIFICATION,
    "cdcp_relclass_jl_inv": CLASSIFICATION,
    "cdcp_relclass_context": CLASSIFICATION,
    "cdcp_relclass_rbert": CLASSIFICATION,
    "cdcp_relclass_rbert_jl": CLASSIFICATION,
    "cdcp_relclass_distance": CLASSIFICATION,
    "cdcp_relclass_distance_components": CLASSIFICATION,
    "cdcp_relclass_resnet": CLASSIFICATION,
    "cdcp_relclass_resnet_jl": CLASSIFICATION,
    "cdcp_relclass_resnet_jl_inv": CLASSIFICATION,
}

if __name__ == '__main__':
    # wrkdir = os.getcwd()
    # processor = ArgMinRelClassProcessorCDCPResNetJointLearningInverseLabels()
    # test_docs = processor.get_test_examples("C:\\Users\\Cyrus\\Documents\\UNI\\Postgrad\\TB3\\medical_transformer\\data\\cdcp\\original")

    wrkdir = os.getcwd()
    processor = ArgMinRelClassProcessorCDCP()
    test_docs = processor.get_test_examples("C:\\Users\\Cyrus\\Documents\\UNI\\Postgrad\\TB3\\medical_transformer\\data\\cdcp\\original")