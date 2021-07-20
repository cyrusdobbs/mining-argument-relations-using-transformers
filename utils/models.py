from abc import ABC

import torch
import numpy as np
from torch import nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torchcrf import CRF
from torch.nn import CrossEntropyLoss, MSELoss


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


class BertForMultipleChoiceRC(BertPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)

        # encoder
        self.bert = BertModel(config)

        # multiple choice
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # relation classification
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier2 = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, task=None):

        if len(input_ids) == 2 and task is not None:
            input_ids, input_ids_rel = input_ids
            token_type_ids, token_type_ids_rel = token_type_ids
            attention_mask, attention_mask_rel = attention_mask
            labels, labels_rel = labels

            # relation classification (required only for training)
            _, pooled_output_rel = self.bert(input_ids_rel, token_type_ids_rel, attention_mask_rel)
            pooled_output_rel = self.dropout2(pooled_output_rel)
            logits_rel = self.classifier2(pooled_output_rel)

        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        _, pooled_output_mc = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask)
        pooled_output_mc = self.dropout(pooled_output_mc)
        logits = self.classifier(pooled_output_mc)
        reshaped_logits = logits.view(-1, num_choices)

        if labels is not None:

            loss_fct = CrossEntropyLoss()

            if task == "multiplechoice":
                loss = loss_fct(reshaped_logits, labels[:, 0])

            elif task == "relationclassification":
                loss = loss_fct(logits_rel, labels_rel)

            return loss, reshaped_logits, logits_rel

        else:

            mc_preds = np.argmax(reshaped_logits, axis=1).flatten()
            input_ids_rel = []
            token_type_ids_rel = []
            attention_mask_rel = []
            for i, pred in enumerate(mc_preds):
                input_ids_rel.append(input_ids[i][pred].unsqueeze(0))
                token_type_ids_rel.append(token_type_ids[i][pred].unsqueeze(0))
                attention_mask_rel.append(attention_mask[i][pred].unsqueeze(0))

            input_ids_rel = torch.cat(input_ids_rel)
            token_type_ids_rel = torch.cat(token_type_ids_rel)
            attention_mask_rel = torch.cat(attention_mask_rel)

            # relation classification
            _, pooled_output_rel = self.bert(input_ids_rel, token_type_ids_rel, attention_mask_rel)
            pooled_output_rel = self.dropout2(pooled_output_rel)
            logits_rel = self.classifier2(pooled_output_rel)

            return logits_rel, reshaped_logits


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
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

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

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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

        # concat_distance = torch.cat([pooled_output, distance_encoding.float()], dim=1)
        # concat_components = torch.cat([concat_distance, component_encoding.float()], dim=1)
        concat_components = torch.cat([text, distance, components], dim=1)  # 384

        logits = self.classifier(concat_components)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)