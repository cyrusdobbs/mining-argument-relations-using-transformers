from transformers import BertModel, RobertaModel, RobertaConfig, RobertaTokenizer

# model = BertModel.from_pretrained("bert-base-uncased")

model = RobertaModel.from_pretrained("roberta-base")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
config = RobertaConfig.from_pretrained("roberta-base")

print("DONE")
