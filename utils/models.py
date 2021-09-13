from abc import ABC

import torch
import numpy as np
from torch import nn
from transformers import RobertaModel
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torchcrf import CRF
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_roberta import RobertaClassificationHead


class BertForSequenceTagging(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.rnn = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        # self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)

        self.crf = CRF(config.num_labels, batch_first=True)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        rnn_out, _ = self.rnn(sequence_output)
        emissions = self.classifier(rnn_out)

        if labels is not None:
            loss = self.crf(emissions, labels)

            path = self.crf.decode(emissions)
            path = torch.LongTensor(path)

            return (-1 * loss, emissions, path)
        else:
            path = self.crf.decode(emissions)
            path = torch.LongTensor(path)

            return path

class BertForSequenceClassificationCDCP(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super(BertForSequenceClassificationCDCP, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = get_classifier(kwargs['classifier_type'], input_size=config.hidden_size, config=config)

        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(kwargs['loss_weights']))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = 100 * self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForSequenceClassificationCDCPJointLearning(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super(BertForSequenceClassificationCDCPJointLearning, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.relation = Softmax(config.hidden_size, self.num_labels)
        self.link = link
        self.source = Softmax(config.hidden_size, 5)
        self.target = Softmax(config.hidden_size, 5)

        loss_weights = kwargs['loss_weights'] if self.num_labels == 3 else [kwargs['loss_weights'][0],
                                                                            kwargs['loss_weights'][1],
                                                                            kwargs['loss_weights'][1] / 5,
                                                                            kwargs['loss_weights'][2],
                                                                            kwargs['loss_weights'][2] / 5]

        self.relation_loss_fct = CrossEntropyLoss(weight=torch.tensor(loss_weights))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, relation_labels=None,
                link_labels=None, source_labels=None, target_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        relation_logits = self.relation(pooled_output)
        # link_logits = self.link(final_output)
        source_logits = self.source(pooled_output)
        target_logits = self.target(pooled_output)

        outputs = (relation_logits, source_logits, target_logits) + outputs[
                                                                    2:]  # add hidden states and attention if they are here

        loss_fct = CrossEntropyLoss()
        relation_loss = self.relation_loss_fct(relation_logits.view(-1, self.num_labels), relation_labels.view(-1))
        # link_loss = loss_fct(link_logits.view(-1, self.num_labels), link_labels.view(-1))
        source_loss = loss_fct(source_logits.view(-1, 5), source_labels.view(-1))
        target_loss = loss_fct(target_logits.view(-1, 5), target_labels.view(-1))

        rw = 100  # Relation loss weight
        lw = 0  # Link loss weight
        sw = 10  # Source loss weight
        tw = 10  # Target loss weight

        # + (lw * link_loss)
        loss = (rw * relation_loss) + (sw * source_loss) + (tw * target_loss)

        # weight decay added via scheduler
        # + torch.norm(self.res_net.first_linear.weight, p=2) + torch.norm(self.res_net.final_linear.weight, p=2)

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForSequenceClassificationCDCP(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super(RobertaForSequenceClassificationCDCP, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(kwargs['loss_weights']))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                labels=None):

        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        loss = 100 * self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super(RBERT, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, config.hidden_dropout_prob)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, config.hidden_dropout_prob)
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            config.hidden_dropout_prob,
            use_activation=False,
        )

        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(kwargs['loss_weights']))

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        loss = 100 * self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RBERTJointLearning(RBERT):
    def __init__(self, config, **kwargs):
        super(RBERTJointLearning, self).__init__(config, **kwargs)

        self.source_classifier = FCLayer(
            config.hidden_size * 3,
            5,
            config.hidden_dropout_prob,
            use_activation=False,
        )

        self.target_classifier = FCLayer(
            config.hidden_size * 3,
            5,
            config.hidden_dropout_prob,
            use_activation=False,
        )

        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(kwargs['loss_weights']))

    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, relation_labels, source_labels,
                target_labels):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)

        relation_logits = self.label_classifier(concat_h)
        source_logits = self.source_classifier(concat_h)
        target_logits = self.target_classifier(concat_h)

        outputs = (relation_logits, source_logits, target_logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        loss_fct = CrossEntropyLoss()
        relation_loss = self.loss_fct(relation_logits.view(-1, self.num_labels), relation_labels.view(-1))
        source_loss = loss_fct(source_logits.view(-1, 5), source_labels.view(-1))
        target_loss = loss_fct(target_logits.view(-1, 5), target_labels.view(-1))

        rw = 100  # Relation loss weight
        sw = 10  # Source loss weight
        tw = 10  # Target loss weight

        loss = (rw * relation_loss) + (sw * source_loss) + (tw * target_loss)

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RBERTJointLearningV2(RBERT):
    def __init__(self, config, **kwargs):
        super(RBERTJointLearningV2, self).__init__(config, **kwargs)

        self.source_classifier = FCLayer(
            config.hidden_size,
            5,
            config.hidden_dropout_prob,
            use_activation=False,
        )

        self.target_classifier = FCLayer(
            config.hidden_size,
            5,
            config.hidden_dropout_prob,
            use_activation=False,
        )

        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(kwargs['loss_weights']))
        self.final_evaluation = kwargs['final_evaluation']

    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, relation_labels, source_labels,
                target_labels):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)

        relation_logits = self.label_classifier(concat_h)
        source_logits = self.source_classifier(e1_h)
        target_logits = self.target_classifier(e2_h)

        outputs = (relation_logits, source_logits, target_logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        loss_fct = CrossEntropyLoss()
        relation_loss = self.loss_fct(relation_logits.view(-1, self.num_labels), relation_labels.view(-1))
        source_loss = loss_fct(source_logits.view(-1, 5), source_labels.view(-1))
        target_loss = loss_fct(target_logits.view(-1, 5), target_labels.view(-1))

        rw = 100  # Relation loss weight
        sw = 10  # Source loss weight
        tw = 10  # Target loss weight

        loss = (rw * relation_loss) + (sw * source_loss) + (tw * target_loss)

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def get_classifier(classifier_type, input_size, config, **kwargs):
    if classifier_type == "2FC":
        return nn.Sequential(nn.Linear(input_size, 256),
                             nn.Linear(256, config.num_labels))
    if classifier_type == "3FC":
        return nn.Sequential(nn.Linear(input_size, 256),
                             nn.Linear(256, 64),
                             nn.Linear(64, config.num_labels))
    elif classifier_type == "FC":
        return nn.Linear(input_size, config.num_labels)
    elif classifier_type == "2FC-RelU":
        return nn.Sequential(nn.Linear(input_size, 256),
                             nn.ReLU(),
                             nn.Linear(256, config.num_labels))
    elif classifier_type == "2FC-Tanh":
        return nn.Sequential(nn.Linear(input_size, 256),
                             nn.Tanh(),
                             nn.Linear(256, config.num_labels))
    elif classifier_type == "FC-DO":
        return FCLayer(input_size, config.num_labels, config.hidden_dropout_prob if config.hidden_dropout_prob else 0.0,
                       use_activation=False)
    else:
        raise Exception("Valid classifier_type param not given.")


class BertForSequenceClassificationWithDistance(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config, **kwargs):
        super(BertForSequenceClassificationWithDistance, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = get_classifier(kwargs['classifier_type'], input_size=config.hidden_size + 10, config=config)

        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(kwargs['loss_weights']))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, distance_encoding=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        concat_distance = torch.cat([pooled_output, distance_encoding.float()], dim=1)
        logits = self.classifier(concat_distance)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = 100 * self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForSequenceClassificationWithDistanceAndComponentType(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config, **kwargs):
        super(BertForSequenceClassificationWithDistanceAndComponentType, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = get_classifier(kwargs['classifier_type'], config.hidden_size + 10 + 8, config)

        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(kwargs['loss_weights']))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                distance_encoding=None, component_encoding=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        concat_distance = torch.cat([pooled_output, distance_encoding.float()], dim=1)
        concat_components = torch.cat([concat_distance, component_encoding.float()], dim=1)

        logits = self.classifier(concat_components)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = 100 * self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForSequenceClassificationWithDistanceAndComponentTypeNew(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config, **kwargs):
        super(BertForSequenceClassificationWithDistanceAndComponentTypeNew, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.fc1 = nn.Linear(config.hidden_size, 128)
        self.fc2 = nn.Linear(10, 128)
        self.fc3 = nn.Linear(8, 128)

        self.classifier = get_classifier(kwargs['classifier_type'], 384, config)

        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(kwargs['loss_weights']))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                distance_encoding=None, component_encoding=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        text = self.fc1(pooled_output)  # to 128
        distance = self.fc2(distance_encoding.float())  # 128
        components = self.fc3(component_encoding.float())  # 128

        concat_components = torch.cat([text, distance, components], dim=1)  # 384

        logits = self.classifier(concat_components)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = 100 * self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class ResidualNetwork(nn.Module):

    def __init__(self, dropout_rate=0.1, res_size=5):
        super(ResidualNetwork, self).__init__()

        self.first_bn = nn.BatchNorm1d(20, eps=0.001, momentum=0.99)
        self.final_bn = nn.BatchNorm1d(5, eps=0.001, momentum=0.99)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        self.first_linear = nn.Linear(in_features=20, out_features=res_size)
        nn.init.kaiming_normal_(self.first_linear.weight)  # He initialisation of weights

        self.final_linear = nn.Linear(in_features=res_size, out_features=20)
        nn.init.kaiming_normal_(self.final_linear.weight)  # He initialisation of weights

    def forward(self, x):
        identity = x

        # Layer 1
        x = self.first_bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.first_linear(x)

        # Layer 2
        x = self.final_bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.final_linear(x)

        x += identity

        return x


class Softmax(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.linear(x)
        return self.softmax(x)

def create_crop_fn(dimension, start, end):
    """
    From https://github.com/keras-team/keras/issues/890#issuecomment-319671916
    Crops (or slices) a Tensor on a given dimension from start to end
    example : to crop tensor x[:, :, 5:10]
    call slice(2, 5, 10) as you want to crop on the second dimension
    :param dimension: dimension of the object. The crop will be performed on the last dimension
    :param start: starting index
    :param end: ending index (excluded)
    :return:
    """
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    func.__name__ = "crop_" + str(dimension) + "_" + str(start) + "_" + str(end)
    return func

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

def link(rel_ol):
    outputs = (2, 5, 5, 5)
    link_as_sum = [[0, 2], [1, 3, 4]]

    link_scores = []
    rel_scores = []
    # creates a layer that extracts the score of a single relation classification class
    for i in range(outputs[1]):
        lam = Lambda(create_crop_fn(1, i, i + 1))
        rel_scores.append(lam.forward(rel_ol))
    #  link_as_a_sum = [[0, 2], [1, 3, 4]]
    #  outputs = (2, 5, 5, 5)
    # for each link class, sums the relation score contributions
    for i in range(len(link_as_sum)):
        # terms to be summed together for one of the link classes
        link_contribute = []
        for j in range(len(link_as_sum[i])):
            value = link_as_sum[i][j]
            link_contribute.append(rel_scores[value])
        link_class = torch.sum(torch.stack(link_contribute), -2)
        link_scores.append(link_class)

    link_ol = torch.cat(link_scores)

    return link_ol


class BertForSequenceClassificationResNet(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config, **kwargs):
        super(BertForSequenceClassificationResNet, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.bert_dense = nn.Linear(in_features=768, out_features=100)

        self.merge_bn = nn.BatchNorm1d(110, eps=0.001, momentum=0.99)
        self.merge_dropout = nn.Dropout(0.1)
        self.merge_dense = nn.Linear(in_features=110, out_features=20)
        nn.init.kaiming_normal_(self.merge_dense.weight)  # He initialisation of weights

        self.res_net = ResidualNetwork()

        self.final_bn = nn.BatchNorm1d(20, eps=0.001, momentum=0.99)
        self.final_dropout = nn.Dropout(0.1)

        self.relation = Softmax(20, 3)

        self.loss_fct = CrossEntropyLoss(weight=torch.tensor(kwargs['loss_weights']))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, distance_encoding=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        reduced_bert = self.bert_dense(pooled_output)

        bert_distance_concat = torch.cat([reduced_bert, distance_encoding.float()], dim=1)  # 110

        pre_resnet = self.merge_bn(bert_distance_concat)
        pre_resnet = self.merge_dropout(pre_resnet)
        pre_resnet = self.merge_dense(pre_resnet)

        resnet_output = self.res_net(pre_resnet)

        final_output = self.final_bn(resnet_output)
        final_output = self.final_dropout(final_output)

        relation_logits = self.relation(final_output)

        outputs = (relation_logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = 100 * self.loss_fct(relation_logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForSequenceClassificationResNetJointLearning(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super(BertForSequenceClassificationResNetJointLearning, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.bert_dense = nn.Linear(in_features=768, out_features=100)

        self.merge_bn = nn.BatchNorm1d(110, eps=0.001, momentum=0.99)
        self.merge_dropout = nn.Dropout(0.1)
        self.merge_dense = nn.Linear(in_features=110, out_features=20)
        nn.init.kaiming_normal_(self.merge_dense.weight)  # He initialisation of weights

        self.res_net = ResidualNetwork()

        self.final_bn = nn.BatchNorm1d(20, eps=0.001, momentum=0.99)
        self.final_dropout = nn.Dropout(0.1)

        self.relation = Softmax(20, self.num_labels)
        self.link = link
        self.source = Softmax(20, 5)
        self.target = Softmax(20, 5)

        loss_weights = kwargs['loss_weights'] if self.num_labels == 3 else [kwargs['loss_weights'][0],
                                                                            kwargs['loss_weights'][1],
                                                                            kwargs['loss_weights'][1] / 5,
                                                                            kwargs['loss_weights'][2],
                                                                            kwargs['loss_weights'][2] / 5]

        self.relation_loss_fct = CrossEntropyLoss(weight=torch.tensor(loss_weights))

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, distance_encoding=None, relation_labels=None,
                link_labels=None, source_labels=None, target_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        reduced_bert = self.bert_dense(pooled_output)

        bert_distance_concat = torch.cat([reduced_bert, distance_encoding.float()], dim=1)  # 110

        pre_resnet = self.merge_bn(bert_distance_concat)
        pre_resnet = self.merge_dropout(pre_resnet)
        pre_resnet = self.merge_dense(pre_resnet)

        resnet_output = self.res_net(pre_resnet)

        final_output = self.final_bn(resnet_output)
        final_output = self.final_dropout(final_output)

        relation_logits = self.relation(final_output)
        # link_logits = self.link(final_output)
        source_logits = self.source(final_output)
        target_logits = self.target(final_output)

        outputs = (relation_logits, source_logits, target_logits) + outputs[2:]  # add hidden states and attention if they are here

        loss_fct = CrossEntropyLoss()
        relation_loss = self.relation_loss_fct(relation_logits.view(-1, self.num_labels), relation_labels.view(-1))
        # link_loss = loss_fct(link_logits.view(-1, self.num_labels), link_labels.view(-1))
        source_loss = loss_fct(source_logits.view(-1, 5), source_labels.view(-1))
        target_loss = loss_fct(target_logits.view(-1, 5), target_labels.view(-1))

        rw = 100  # Relation loss weight
        lw = 0  # Link loss weight
        sw = 10  # Source loss weight
        tw = 10  # Target loss weight

        # + (lw * link_loss)
        loss = (rw * relation_loss) + (sw * source_loss) + (tw * target_loss)

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

# class BertForMultipleChoiceRC(BertPreTrainedModel):
#
#     def __init__(self, config, **kwargs):
#         super().__init__(config)
#
#         # encoder
#         self.bert = BertModel(config)
#
#         # multiple choice
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#
#         # relation classification
#         self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier2 = nn.Linear(config.hidden_size, 2)
#
#         self.init_weights()
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, task=None):
#
#         if len(input_ids) == 2 and task is not None:
#             input_ids, input_ids_rel = input_ids
#             token_type_ids, token_type_ids_rel = token_type_ids
#             attention_mask, attention_mask_rel = attention_mask
#             labels, labels_rel = labels
#
#             # relation classification (required only for training)
#             _, pooled_output_rel = self.bert(input_ids_rel, token_type_ids_rel, attention_mask_rel)
#             pooled_output_rel = self.dropout2(pooled_output_rel)
#             logits_rel = self.classifier2(pooled_output_rel)
#
#         num_choices = input_ids.shape[1]
#
#         flat_input_ids = input_ids.view(-1, input_ids.size(-1))
#         flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
#         flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#
#         _, pooled_output_mc = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask)
#         pooled_output_mc = self.dropout(pooled_output_mc)
#         logits = self.classifier(pooled_output_mc)
#         reshaped_logits = logits.view(-1, num_choices)
#
#         if labels is not None:
#
#             loss_fct = CrossEntropyLoss()
#
#             if task == "multiplechoice":
#                 loss = loss_fct(reshaped_logits, labels[:, 0])
#
#             elif task == "relationclassification":
#                 loss = loss_fct(logits_rel, labels_rel)
#
#             return loss, reshaped_logits, logits_rel
#
#         else:
#
#             mc_preds = np.argmax(reshaped_logits, axis=1).flatten()
#             input_ids_rel = []
#             token_type_ids_rel = []
#             attention_mask_rel = []
#             for i, pred in enumerate(mc_preds):
#                 input_ids_rel.append(input_ids[i][pred].unsqueeze(0))
#                 token_type_ids_rel.append(token_type_ids[i][pred].unsqueeze(0))
#                 attention_mask_rel.append(attention_mask[i][pred].unsqueeze(0))
#
#             input_ids_rel = torch.cat(input_ids_rel)
#             token_type_ids_rel = torch.cat(token_type_ids_rel)
#             attention_mask_rel = torch.cat(attention_mask_rel)
#
#             # relation classification
#             _, pooled_output_rel = self.bert(input_ids_rel, token_type_ids_rel, attention_mask_rel)
#             pooled_output_rel = self.dropout2(pooled_output_rel)
#             logits_rel = self.classifier2(pooled_output_rel)
#
#             return logits_rel, reshaped_logits