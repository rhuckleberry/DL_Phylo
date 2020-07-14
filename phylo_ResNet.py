#!/usr/bin/env python3

"""Quartet tree classification

* Model: Convolutional neural network with basic residual connections
  and batch normalization.
* Training data:
    * 100000 pre-simulated trees using training1.
    * Each epoch uses randomly sampled 2000 trees.
    * The batch size is 16.
* Validation data: 2000 pre-simulated trees using training1.
* Optimizer: Adam with an initial learning rate of 0.001.
"""
print("imports")
import visdom
import pathlib
import pickle
import random

import numpy as np
import torch.autograd
import torch.nn
import torch.optim
import torch.utils.data


class _Model(torch.nn.Module):
    """A neural network model to predict phylogenetic trees."""

    def __init__(self):
        """Create a neural network model."""
        print("making model")
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(16, 80, 1, groups=4),
            torch.nn.BatchNorm1d(80),
            torch.nn.ReLU(),
            torch.nn.Conv1d(80, 32, 1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AvgPool1d(2),
            _ResidueModule(32),
            _ResidueModule(32),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = torch.nn.Linear(32, 6)

    def forward(self, x):
        """Predict phylogenetic trees for the given sequences.

        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequences.

        Returns
        -------
        torch.Tensor
            The predicted adjacency trees.
        """

        #print("forward")
        x = x.view(x.size()[0], 16, -1)
        x = self.conv(x).squeeze(dim=2)
        return self.classifier(x)


class _ResidueModule(torch.nn.Module):

    def __init__(self, channel_count):
        #print("making resnet")
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(channel_count, channel_count, 1),
            torch.nn.BatchNorm1d(channel_count),
            torch.nn.ReLU(),
            torch.nn.Conv1d(channel_count, channel_count, 1),
            torch.nn.BatchNorm1d(channel_count),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        #print("forward resnet")
        return x + self.layers(x)

def permute_data(datapoint):

    tree_type = datapoint[1]
    permuted_datapoints = None

    if tree_type == 0: #alpha
        permuted_datapoints = permute._alpha_permute(datapoint)
    elif tree_type == 1: #beta
        permuted_datapoints = permute._beta_permute(datapoint)
    elif tree_type == 2: #gamma
        permuted_datapoints = permute._gamma_permute(datapoint)
    else:
        print("Error: tree type not defined")

    random.shuffle(permuted_datapoints)

    return permuted_datapoints

training_data = np.load("seq-gen/data/train/training_data.npy", allow_pickle = True)
dev_data = np.load("seq-gen/data/dev/development_data.npy", allow_pickle = True)
train_data = training_data.tolist()
validation_data = dev_data.tolist()
print(len(train_data))
print(len(validation_data))
print("loaded data")

#plotting
vis = visdom.Visdom()

#model Hyperparameters
model = _Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#weight initialization...
loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

BATCH_SIZE = 16
TRAIN_SIZE = 3600
epoch = 1

#Train
while epoch < 300:

    #TRAIN
    model.train()
    #randomly take 2000 datapoints
    epoch_train = random.sample(train_data, TRAIN_SIZE)
    sample_count, correct, score = 0, 0, 0.0

    for i in range(TRAIN_SIZE // BATCH_SIZE):
        data = epoch_train[i * BATCH_SIZE : (i+1) * BATCH_SIZE]

        x_list = []
        y_list = []

        # transformed_data = []
        # for datapoint in data:
        #     all_datapoint = permute_data(datapoint)
        #     transformed_data += all_datapoint
        #
        # random.shuffle(transformed_data)

        for datapoint in data: #transformed_data:
            sequences = datapoint[0]
            label = datapoint[1]
            x_list.append(sequences)
            y_list.append(label)


        x = torch.tensor(x_list, dtype=torch.float)
        x = x.view(BATCH_SIZE, 4, 4, -1)
        y = torch.tensor(y_list)
        sample_count += x.size()[0]

        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        score += float(loss)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == y).sum().item()

        print("\n")
        #print(output)
        print(predicted)
        print(y)
        print("\n")

    score /= sample_count
    accuracy = correct / sample_count
    print("\n", "\n")
    print(f'Epoch{epoch}')
    print("\n")
    print("Training:")
    print(f'Training loss: {score}')
    print(f'Accuracy: {accuracy}')

    vis.line(
        X = [epoch],
        Y = [accuracy],
        opts= dict(title="ResNet Tree Model 8",
               xlabel="Epochs",
               showlegend=True),
        win= "graph8",
        name = "Train Accuracy",
        update="append"
        )
    vis.line(
        X = [epoch],
        Y = [score],
        win= "graph8",
        name = "Train Score",
        update="append"
        )

    ##VALIDATE
    optimizer.zero_grad()
    model.train(False)
    sample_count, correct, score = 0, 0, 0.0

    tree_0_len, tree_1_len, tree_2_len = 0, 0, 0
    guess_0, guess_1, guess_2 = 0,0,0
    real_0, real_1, real_2 = 0,0,0

    # ##PERMUTE
    # for data in validation_data:
    #
    #     transformed_data = transform(data)
    #     print(len(transformed_data))
    #     print([i for _, i in transformed_data])
    #
    #     x_list = []
    #     y_list = []
    #
    #     for datapoint in transformed_data:
    #         sequences = datapoint[0]
    #         label = datapoint[1]
    #         x_list.append(sequences)
    #         y_list.append(label)
    #
    #     x = torch.tensor(x_list, dtype=torch.float)
    #     x = x.view(24, 4, 4, -1)
    #     y = torch.tensor(y_list)
    #     sample_count += x.size()[0]
    #
    #     output = model(x)
    #     loss = loss_function(output, y)
    #
    #     score += float(loss)
    #     _, predicted = torch.max(output.data, 1)
    #     correct += (predicted == y).sum().item()




    # #NO PERMUTE
    # DEV_BATCH_SIZE = BATCH_SIZE * 24
    # DEV_SIZE = len(validation_data)
    #
    # for i in range(DEV_SIZE // DEV_BATCH_SIZE):
    #     data = validation_data[i * DEV_BATCH_SIZE : (i+1) * DEV_BATCH_SIZE]
    #
    #     x_list = []
    #     y_list = []
    #
    #     for datapoint in data:
    #         sequences = datapoint[0]
    #         label = datapoint[1]
    #         x_list.append(sequences)
    #         y_list.append(label)
    #
    #     x = torch.tensor(x_list, dtype=torch.float)
    #     x = x.view(DEV_BATCH_SIZE, 4, 4, -1)
    #     y = torch.tensor(y_list)
    #     sample_count += x.size()[0]
    #
    #     output = model(x)
    #     loss = loss_function(output, y)
    #
    #     score += float(loss)
    #     _, predicted = torch.max(output.data, 1)
    #     correct += (predicted == y).sum().item()


    #NO PERMUTE -- batch size of 1
    for x, y in validation_data:

        x = torch.tensor(x, dtype=torch.float)
        x = x.view(1, 4, 4, -1)
        y = torch.tensor([y])
        sample_count += x.size()[0]

        output = model(x)
        loss = loss_function(output, y)

        score += float(loss)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == y).sum().item()

        wrong = [(predicted[i], i) for i in range(len(predicted)) if predicted[i] != y[i]]
        tree_0 = [tree for tree, i in wrong if y[i] == 0]
        tree_1 = [tree for tree, i in wrong if y[i] == 1]
        tree_2 = [tree for tree, i in wrong if y[i] == 2]
        print("\n", "\n")
        print(predicted)
        print(y)
        print("\n")
        print(wrong, len(wrong))
        print("\n", "\n")

        tree_0_len += len(tree_0)
        tree_1_len += len(tree_1)
        tree_2_len += len(tree_2)

        guess_0 += len([i for i in predicted if i == 0])
        guess_1 += len([i for i in predicted if i == 1])
        guess_2 += len([i for i in predicted if i == 2])

        real_0 += len([i for i in y if i == 0])
        real_1 += len([i for i in y if i == 1])
        real_2 += len([i for i in y if i == 2])

    print("done")

    score /= sample_count
    accuracy = correct / sample_count
    print("\n", "\n")
    print("Validation:")
    print(f'Validation loss: {score}')
    print(f'Accuracy: {accuracy}')
    print("\n", "\n")
    #
    # vis.bar(
    #     X = [tree_0_len, tree_1_len, tree_2_len, guess_0, guess_1, guess_2,
    #          real_0, real_1, real_2],
    #     opts= dict(title= f"Prediction Fails {epoch}",
    #            rownames=["tree_0", "tree_1", "tree_2", "guess_0", "guess_1", "guess_2",
    #                      "real_0", "real_1", "real_2"]),
    #            win= f"bar3{epoch}"
    # )

    vis.line(
        X = [epoch],
        Y = [accuracy],
        win= "graph8",
        name = "Dev Accuracy",
        update="append"
        )
    vis.line(
        X = [epoch],
        Y = [score],
        win= "graph8",
        name = "Dev Score",
        update="append"
        )

    epoch += 1
