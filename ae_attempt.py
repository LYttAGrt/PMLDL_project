import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np

import itertools

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


# All files are expected to be in same folder
def parse_data(folder_path='anecdots', files_cnt=1):
    parsed_values: list = []
    cnt = 1
    for each in os.listdir(folder_path):
        with open(folder_path + '/' + each, 'r') as f:
            buf = pd.read_csv(folder_path + '/' + each, sep=',')
            parsed_values += buf['content'].tolist()
        if cnt >= files_cnt:
            break
        else:
            cnt += 1
    return parsed_values


class AE(nn.Module):
    def __init__(self, shapes: tuple):
        super().__init__()
        # Encoder
        self.line1 = nn.Linear(in_features=shapes[0], out_features=shapes[1])
        self.line2 = nn.Linear(in_features=shapes[1], out_features=shapes[2])

        # Decoder
        self.line3 = nn.Linear(in_features=shapes[2], out_features=shapes[1])
        self.line4 = nn.Linear(in_features=shapes[1], out_features=shapes[0])

    def forward(self, data: torch.Tensor):
        # Encode
        z = F.relu(self.line1(data))
        z = self.line2(z)

        # Decode
        z = self.line3(z)
        z = self.line4(z)
        return z


# data: list of words in 2d
def idx_data(data: list):
    lookup = sorted(list(set(itertools.chain.from_iterable([sentence_data for sentence_data in data]))))
    lookup = {value: index for index, value in enumerate(lookup, 1)}
    return lookup, {index: value for index, value in enumerate(lookup, 1)}


def coalesce(*inputs):
    for i in range(len(inputs)):
        if inputs[i] is not None:
            return inputs[i]
    return 0


# 1D list of sentences
def preprocess(text: list) -> (torch.Tensor, dict, dict):
    # Tokenize all sentences to words. Format is 2D: <sentence, word>
    tokenized_dataset = list()
    for joke in text:
        tokenized_dataset.append(nltk.tokenize.word_tokenize(joke, language='russian'))

    # Drop tail (optional)
    tokenized_dataset = tokenized_dataset[:len(tokenized_dataset) - len(tokenized_dataset) % batch_size]

    # Convert tokens to vectors using Word2Vec
    word_to_idx, ids_to_word = idx_data(tokenized_dataset)
    indexes = []
    for sentence in tokenized_dataset:
        indexes.append([coalesce(word_to_idx.get(word)) for word in sentence])

    # Pad to 2D matrix
    max_line_len = len(max(tokenized_dataset, key=len))
    tensor = torch.zeros(size=(len(text), max_line_len))
    for i in range(len(indexes)):
        for j in range(len(indexes[i])):
            tensor[i, j] = indexes[i][j]

    return tensor, word_to_idx, ids_to_word


if __name__ == '__main__':
    # Get some data for model. We have Russian jokes.
    batch_size = 32
    rus_data = parse_data()

    # <cnt of lines, cnt of words>
    dataset, direct_lookup, reverse_lookup = preprocess(rus_data)
    dataset /= len(direct_lookup)

    device = torch.device('cpu')
    model = AE(shapes=(dataset.shape[1], dataset.shape[1] // 2, dataset.shape[1] // 4))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    loss = nn.CrossEntropyLoss()
    loss.to(device)

    # Train
    model.train()
    prev_sum = .0
    for epoch in range(10):
        loss_sum = .0
        for batch in range(len(dataset) // batch_size):
            optimizer.zero_grad()
            output = model(dataset[batch_size*batch:(batch+1)*batch_size, :])
            # Generate labels {as True, False}
            labels = torch.sign(torch.round(torch.abs(torch.sum(output - dataset[batch_size*batch:(batch+1)*batch_size, :], dim=1))))

            loss_res = loss(output, labels.long())
            loss_sum += loss_res
            loss_res.backward()
            optimizer.step()
        print(epoch, loss_sum.item())

    # Evaluate
    print(f'Length of direct lookup: {len(direct_lookup)}')
    model.eval()
    testing: torch.Tensor = model(dataset[2, :])
    values: np.ndarray = np.round(testing.detach().numpy())
    print(rus_data[2], '\n', dataset[2, :].numpy() * len(direct_lookup), '->', values[:])
    print(' '.join([
        reverse_lookup.get(idx) for idx in values.tolist()[:values.argmin()] if idx in reverse_lookup.keys()
    ]).lower())
