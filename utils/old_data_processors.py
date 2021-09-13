# import collections
# import os
#
# from utils.data_processors import DataProcessor, InputExample
#
#
# class MultiChoiceExample(object):
#     """A single training/test example for the SWAG dataset."""
#
#     def __init__(self,
#                  swag_id,
#                  context_sentence,
#                  ending_0,
#                  ending_1,
#                  ending_2,
#                  ending_3,
#                  ending_4,
#                  ending_5,
#                  label=None):
#         self.swag_id = swag_id
#         self.context_sentence = context_sentence
#         self.endings = [
#             ending_0,
#             ending_1,
#             ending_2,
#             ending_3,
#             ending_4,
#             ending_5,
#         ]
#         self.label = label
#
#     def __str__(self):
#         return self.__repr__()
#
#     def __repr__(self):
#         l = [
#             "swag_id: {}".format(self.swag_id),
#             "context_sentence: {}".format(self.context_sentence),
#             "ending_0: {}".format(self.endings[0]),
#             "ending_1: {}".format(self.endings[1]),
#             "ending_2: {}".format(self.endings[2]),
#             "ending_3: {}".format(self.endings[3]),
#             "ending_4: {}".format(self.endings[4]),
#             "ending_5: {}".format(self.endings[5]),
#         ]
#
#         if self.label is not None:
#             l.append("label: {}".format(self.label))
#
#         return ", ".join(l)
#
#     @classmethod
#     def truncate_seq_pair(cls, tokens_a, tokens_b, max_length):
#         """Truncates a sequence pair in place to the maximum length."""
#
#         # This is a simple heuristic which will always truncate the longer sequence
#         # one token at a time. This makes more sense than truncating an equal percent
#         # of tokens from each, since if one sequence is very short then each token
#         # that's truncated likely contains more information than a longer sequence.
#         while True:
#             total_length = len(tokens_a) + len(tokens_b)
#             if total_length <= max_length:
#                 break
#             if len(tokens_a) > len(tokens_b):
#                 tokens_a.pop()
#             else:
#                 tokens_b.pop()
#
#
# class MultiChoiceInputFeatures(object):
#     def __init__(self,
#                  example_id,
#                  choices_features,
#                  label
#
#                  ):
#         self.example_id = example_id
#         self.choices_features = [
#             {
#                 'input_ids': input_ids,
#                 'input_mask': input_mask,
#                 'segment_ids': segment_ids
#             }
#             for _, input_ids, input_mask, segment_ids in choices_features
#         ]
#         self.label = label
#
#     @classmethod
#     def select_field(cls, features, field):
#         return [
#             [
#                 choice[field]
#                 for choice in feature.choices_features
#             ]
#             for feature in features
#         ]
#
# class ArgMinRelClassProcessor(DataProcessor):
#     """Processor for the RCT data set (for training)."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, "train_relations.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, "dev_relations.tsv")), "dev")
#
#     def get_test_examples(self, data_dir, setname="test_relations.tsv"):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, setname)), "test")
#
#     def get_labels(self):
#         """See base class."""
#         return ["__label__noRel", "__label__Support", "__label__Attack"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             # skip first line (e.g. PE dataset)
#             # if i == 0:
#             #    continue
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[1]
#             text_b = line[2]
#             label = line[0]
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=label))
#         return examples
#
#
# class ArgMinSeqTagProcessor(DataProcessor):
#     """Processor for RCT data set (CoNLL format)"""
#
#     def __init__(self):
#         self.labels = ["X", "B-Claim", "I-Claim", "B-Premise", "I-Premise", 'O']
#         self.label_map = self._create_label_map()
#         self.replace_labels = {
#             'B-MajorClaim': 'B-Claim',
#             'I-MajorClaim': 'I-Claim',
#         }
#
#     def _create_label_map(self):
#         label_map = collections.OrderedDict()
#         for i, label in enumerate(self.labels):
#             label_map[label] = i
#         return label_map
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_conll(os.path.join(data_dir, "train.conll"), replace=self.replace_labels), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_conll(os.path.join(data_dir, "dev.conll"), replace=self.replace_labels), "dev")
#
#     def get_test_examples(self, data_dir, setname="test.conll"):
#         """See base class."""
#         return self._create_examples(
#             self._read_conll(os.path.join(data_dir, setname), replace=self.replace_labels), "test")
#
#     def get_labels(self):
#         """ See base class."""
#         return self.labels
#
#     def convert_labels_to_ids(self, labels):
#         idx_list = []
#         for label in labels:
#             idx_list.append(self.label_map[label])
#         return idx_list
#
#     def convert_ids_to_labels(self, idx_list):
#         labels_list = []
#         for idx in idx_list:
#             labels_list.append([key for key in self.label_map.keys() if self.label_map[key] == idx][0])
#         return labels_list
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             guid = "%s-%s" % (set_type, str(i))
#             text_a = line[0]
#             labels = line[-1]
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=None, labels=labels))
#         return examples
#
#
# class ArgMinRelClassForMultiChoiceProcessor(ArgMinRelClassProcessor):
#     """Processor for the RCT data set (for the relation classification in the multiple choice training)."""
#
#     def get_labels(self):
#         """See base class."""
#         return ["__label__Support", "__label__Attack"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#
#             if line[0] == "__label__noRel":
#                 continue
#
#             guid = "%s-%s" % (set_type, line[0])
#             text_a = line[1]
#             text_b = line[2]
#             label = line[0]
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=label))
#         return examples
#
#
# class ArgMinMultiChoiceLinkProcessor(DataProcessor):
#
#     def __init__(self):
#         super().__init__()
#         self.labelmap = {
#             "NoRelation": 2,
#             "Support": 0,
#             "Attack": 1,
#             "Partial-Attack": 1
#         }
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, "train_mc.tsv")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, "dev_mc.tsv")), "dev")
#
#     def get_test_examples(self, data_dir, setname="test_mc.tsv"):
#         """See base class."""
#         return self._create_examples(
#             self._read_tsv(os.path.join(data_dir, setname)), "test")
#
#     def get_labels(self):
#         """ See base class."""
#         return ["0", "1", "2", "3", "4", "5"]
#
#     def _create_examples(self, lines, set_type):
#         examples = []
#
#         for i, line in enumerate(lines):
#             guid = "%s-%s" % (set_type, str(i))
#             context_sentence = line[0]
#             ending_0 = line[1]
#             ending_1 = line[2]
#             ending_2 = line[3]
#             ending_3 = line[4]
#             ending_4 = line[5]
#             ending_5 = line[6]
#             # label = int(line[7])
#             label = (int(line[7]), self.labelmap[line[8]])
#             examples.append(MultiChoiceExample(
#                 swag_id=guid,
#                 context_sentence=context_sentence,
#                 ending_0=ending_0,
#                 ending_1=ending_1,
#                 ending_2=ending_2,
#                 ending_3=ending_3,
#                 ending_4=ending_4,
#                 ending_5=ending_5,
#                 label=label
#             ))
#         return examples
#
#     def convert_examples_to_features(self, examples, tokenizer, max_seq_length, logger=None):
#         """Loads a data file into a list of `InputBatch`s."""
#
#         # Swag is a multiple choice task. To perform this task using Bert,
#         # we will use the formatting proposed in "Improving Language
#         # Understanding by Generative Pre-Training" and suggested by
#         # @jacobdevlin-google in this issue
#         # https://github.com/google-research/bert/issues/38.
#         #
#         # Each choice will correspond to a sample on which we run the
#         # inference. For a given Swag example, we will create the 4
#         # following inputs:
#         # - [CLS] context [SEP] choice_1 [SEP]
#         # - [CLS] context [SEP] choice_2 [SEP]
#         # - [CLS] context [SEP] choice_3 [SEP]
#         # - [CLS] context [SEP] choice_4 [SEP]
#         # The model will output a single value for each input. To get the
#         # final decision of the model, we will run a softmax over these 4
#         # outputs.
#         features = []
#         for example_index, example in enumerate(examples):
#             context_tokens = tokenizer.tokenize(example.context_sentence)
#
#             choices_features = []
#             for ending_index, ending in enumerate(example.endings):
#                 # We create a copy of the context tokens in order to be
#                 # able to shrink it according to ending_tokens
#                 context_tokens_choice = context_tokens[:]
#                 ending_tokens = tokenizer.tokenize(ending)
#                 # Modifies `context_tokens_choice` and `ending_tokens` in
#                 # place so that the total length is less than the
#                 # specified length.  Account for [CLS], [SEP], [SEP] with
#                 # "- 3"
#                 MultiChoiceExample.truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
#
#                 tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
#                 segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)
#
#                 input_ids = tokenizer.convert_tokens_to_ids(tokens)
#                 input_mask = [1] * len(input_ids)
#
#                 # Zero-pad up to the sequence length.
#                 padding = [0] * (max_seq_length - len(input_ids))
#                 input_ids += padding
#                 input_mask += padding
#                 segment_ids += padding
#
#                 assert len(input_ids) == max_seq_length
#                 assert len(input_mask) == max_seq_length
#                 assert len(segment_ids) == max_seq_length
#
#                 choices_features.append((tokens, input_ids, input_mask, segment_ids))
#
#             label = example.label
#             if example_index < 3 and logger is not None:
#                 logger.info("*** Example ***")
#                 logger.info("example_id: {}".format(example.swag_id))
#                 for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
#                     logger.info("choice_idx: {}".format(choice_idx))
#                     logger.info("tokens: {}".format(' '.join(tokens)))
#                     logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
#                     logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
#                     logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
#                     logger.info("label: {}".format(label))
#
#             features.append(
#                 MultiChoiceInputFeatures(
#                     example_id=example.swag_id,
#                     choices_features=choices_features,
#                     label=label
#                 )
#             )
#
#         return features
#
#
# processors = {"seqtag": ArgMinSeqTagProcessor,
#               "relclass": ArgMinRelClassProcessor,
#               "multichoice": (ArgMinRelClassForMultiChoiceProcessor, ArgMinMultiChoiceLinkProcessor)}
#
# output_modes = {"seqtag": "sequence_tagging",
#                 "relclass": "classification",
#                 "multichoice": "classification"}
