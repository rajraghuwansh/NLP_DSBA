import torch
from torch import nn
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from helpers import *


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(
            "activebus/BERT_Review", output_hidden_states=True)
        self.elu1 = nn.ELU()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(EMBEDDING_SIZE+NUM_CATEGORIES, 300)
        self.elu2 = nn.ELU()
        self.drop2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(300, 150)
        self.elu3 = nn.ELU()
        self.drop3 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(150+NUM_CATEGORIES, n_classes)

    def forward(self, input_ids, attention_mask, category_dummies):
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
            ).hidden_states

        num_batches = len(hidden_states[0])
        sentence_embedding = torch.zeros(num_batches, EMBEDDING_SIZE)
        sentence_embedding = sentence_embedding.to(DEVICE)
        for layer in hidden_states[-4:]:
            layer_embedding = torch.mean(layer, dim=1)  # sentence vector of the layer
            sentence_embedding += layer_embedding

        sentence_embedding /= 4  # average sentence vector
        next_input = torch.cat((sentence_embedding, category_dummies), dim=1)
        next_input = self.elu1(next_input)
        next_input = self.drop1(next_input)
        next_input = self.fc1(next_input)
        next_input = self.elu2(next_input)
        next_input = self.drop2(next_input)
        next_input = self.fc2(next_input)
        next_input = self.elu3(next_input)
        next_input = self.drop3(next_input)
        next_input = torch.cat((next_input, category_dummies), dim=1)
        output = self.fc3(next_input)
        return output


class ABSA_Dataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_length):
        self._sentences = data_frame["sentence"]
        self._targets = torch.tensor(data_frame["target"], dtype=torch.long)
        self._category_dummies = data_frame[CATEGORY_NAMES].to_numpy()
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, item):
        text = self._sentences[item]
        target = self._targets[item]
        category_dummies = self._category_dummies[item, :]
        encoded_text = self._tokenizer.encode_plus(
            text,
            return_tensors='pt',
            max_length=self._max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            )

        bert_dict = dict()
        bert_dict["category_dummies"] = category_dummies
        bert_dict["targets"] = target
        bert_dict["input_ids"] = encoded_text["input_ids"][0]
        bert_dict["attention_mask"] = encoded_text["attention_mask"][0]
        return bert_dict
