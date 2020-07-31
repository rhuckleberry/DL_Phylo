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
import copy

import numpy as np
import torch.autograd
import torch.nn
import torch.optim
import torch.utils.data

NUM_MODELS = 8 #number of models to classify

n = 4 #number of nucleotides (amount needed for hot encoding)
s = 4 #number of sequences
c = 200 #sequence length

class _Model(torch.nn.Module):
    """A neural network model to predict phylogenetic trees."""

    def __init__(self):
        """Create a neural network model."""
        print("making model")
        super().__init__()

        k = 32
        l = 64
        #p = 32

        self.base_freq = torch.nn.Sequential(
            torch.nn.Conv2d(s, k, (c,1)),
            torch.nn.BatchNorm2d(k),
            torch.nn.ReLU(),

            _ResidueModule(k),
            _ResidueModule(k),
            # torch.nn.AvgPool2d(kernel_size=(1,2)),
            _ResidueModule(k),
            _ResidueModule(k),
            # torch.nn.AvgPool2d(kernel_size=(1,2)),
            _ResidueModule(k),
            _ResidueModule(k),
            # torch.nn.AvgPool2d(kernel_size=(1,2)),

            torch.nn.Conv2d(k, l, (1,n)),
            torch.nn.BatchNorm2d(l),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            )

        self.rate_mx = torch.nn.Sequential(
            torch.nn.Conv2d(s, k, (1,n)),
            torch.nn.BatchNorm2d(k),
            torch.nn.ReLU(),

            _ResidueModule(k),
            _ResidueModule(k),
            # torch.nn.AvgPool2d(kernel_size=(2,1)),
            _ResidueModule(k),
            _ResidueModule(k),
            # torch.nn.AvgPool2d(kernel_size=(2,1)),
            _ResidueModule(k),
            _ResidueModule(k),
            # torch.nn.AvgPool2d(kernel_size=(2,1)),

            torch.nn.Conv2d(k, l, (c,1)),
            torch.nn.BatchNorm2d(l),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            )

        self.classifier = torch.nn.Linear(2*l, NUM_MODELS)

        # self.res = torch.nn.Sequential(
        #     _ResidueModule(k),
        #     _ResidueModule(k),
        #     torch.nn.AvgPool2d(kernel_size=(3,3), padding=(1,1)),
        #     _ResidueModule(k),
        #     _ResidueModule(k),
        #     torch.nn.AvgPool2d(kernel_size=(3,3), padding=(1,1)),
        #     _ResidueModule(k),
        #     _ResidueModule(k),
        #     torch.nn.AvgPool2d(kernel_size=(3,3), padding=(1,1)),
        #     _ResidueModule(k),
        #     _ResidueModule(k),
        #     torch.nn.AdaptiveAvgPool2d(1)
        # )
        # self.fcc = torch.nn.Linear(l, p)


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
        y = copy.deepcopy(x)

        y = self.base_freq(y)
        x = self.rate_mx(x)

        y = y.view(y.size(0), -1)
        x = x.view(x.size(0), -1)
        z = torch.cat([x,y],1)

        # print("x: ", x.shape)
        # print("y: ", y.shape)
        # print("z: ", z.shape)

        return self.classifier(z)


class _ResidueModule(torch.nn.Module):

    def __init__(self, channel_count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(channel_count, channel_count, (1, 1)),
            torch.nn.BatchNorm2d(channel_count),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel_count, channel_count, (1, 1)),
            torch.nn.BatchNorm2d(channel_count),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layers(x)

def loss_function(outputs, labels, base_weight=100):
    """
    Evolution Model Loss Function - cross entropy + MSE
    """

    cross_entropy = torch.nn.CrossEntropyLoss(reduction='sum')

    loss = 0
    for index, output in enumerate(outputs):

        #cross entropy index
        p1 = torch.tensor([list(output[:2])])
        y1 = torch.tensor([int(labels[index][0])])
        loss += cross_entropy(p1, y1) #* base_weight
        print("base_freq: ", loss)

        #MSE index
        p2 = output[2:]
        y2 = labels[index][1:]
        loss += torch.dot(p2-y2, p2-y2)
        print("rate_mx: ", torch.dot(p2-y2, p2-y2))

    return loss


training_data = np.load("ResNet_data/test_model_train.npy", allow_pickle = True)
dev_data = np.load("ResNet_data/test_model_dev.npy", allow_pickle = True)

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
# loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

BATCH_SIZE = 16
TRAIN_SIZE = len(train_data) // 5
VALIDATION_SIZE = len(validation_data)
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

        for datapoint in data: #transformed_data:
            sequences = datapoint[0]
            label = datapoint[1]
            x_list.append(sequences)
            y_list.append(label)


        x = torch.tensor(x_list, dtype=torch.float)
        x = x.view(BATCH_SIZE, s, c, n)
        y = torch.tensor(y_list)
        sample_count += x.size()[0]

        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        score += float(loss)
        # _, predicted = torch.max(output.data, 1)
        # correct += (predicted == y).sum().item()

        # print("\n")
        # print("output: ", output)
        # # print(predicted)
        # print("label: ", y)
        # print("total: ", sample_count)
        # print("\n")

    score /= sample_count
    # accuracy = correct / sample_count
    # print("\n", "\n")
    # print(f'Epoch{epoch}')
    # print("\n")
    # print("Training:")
    # print(f'Training loss: {score}')
    # # print(f'Accuracy: {accuracy}')

    # vis.line(
    #     X = [epoch],
    #     Y = [accuracy],
    #     opts= dict(title="Inception resnet Model Classifier-Combine Arch- whole more param",
    #            xlabel="Epochs",
    #            showlegend=True),
    #     win= "mod_test11",
    #     name = "Train Accuracy",
    #     update="append"
    #     )
    vis.line(
        X = [epoch],
        Y = [score],
        opts= dict(title="Inception resnet Model Classifier-new way",
                   xlabel="Epochs",
                   showlegend=True),
        win= "mod_test11",
        name = "Train Score",
        update="append"
        )

    ##VALIDATE
    optimizer.zero_grad()
    model.train(False)
    sample_count, correct, score = 0, 0, 0.0

    tree_0_len, tree_1_len, tree_2_len, tree_3_len, tree_4_len, tree_5_len = 0,0,0,0,0,0
    guess_0, guess_1, guess_2, guess_3, guess_4, guess_5 = 0,0,0,0,0,0
    real_0, real_1, real_2, real_3, real_4, real_5 = 0,0,0,0,0,0

    validation_sample = random.sample(validation_data, VALIDATION_SIZE)

    #NO PERMUTE -- batch size of 1
    for x, y in validation_sample:

        x = torch.tensor(x, dtype=torch.float)
        x = x.view(1, s, c, n)
        y = torch.tensor([y])
        sample_count += x.size()[0]

        output = model(x)
        loss = loss_function(output, y)

        score += float(loss)
        # _, predicted = torch.max(output.data, 1)
        # correct += (predicted == y).sum().item()

        # print("\n\n", "output: ", output)
        # print("label: ", y)
        # print("total: ", sample_count, "\n\n")

    #     wrong = [(predicted[i], i) for i in range(len(predicted)) if predicted[i] != y[i]]
    #     tree_0 = [tree for tree, i in wrong if y[i] == 0]
    #     tree_1 = [tree for tree, i in wrong if y[i] == 1]
    #     tree_2 = [tree for tree, i in wrong if y[i] == 2]
    #     tree_3 = [tree for tree, i in wrong if y[i] == 3]
    #     tree_4 = [tree for tree, i in wrong if y[i] == 4]
    #     tree_5 = [tree for tree, i in wrong if y[i] == 5]
    #
    #     print("\n", "\n")
    #     print(predicted)
    #     print(y)
    #     print("\n")
    #     print(wrong, len(wrong))
    #     print("\n", "\n")
    #
    #     tree_0_len += len(tree_0)
    #     tree_1_len += len(tree_1)
    #     tree_2_len += len(tree_2)
    #     tree_3_len += len(tree_3)
    #     tree_4_len += len(tree_4)
    #     tree_5_len += len(tree_5)
    #
    #     guess_0 += len([i for i in predicted if i == 0])
    #     guess_1 += len([i for i in predicted if i == 1])
    #     guess_2 += len([i for i in predicted if i == 2])
    #     guess_3 += len([i for i in predicted if i == 3])
    #     guess_4 += len([i for i in predicted if i == 4])
    #     guess_5 += len([i for i in predicted if i == 5])
    #
    #     real_0 += len([i for i in y if i == 0])
    #     real_1 += len([i for i in y if i == 1])
    #     real_2 += len([i for i in y if i == 2])
    #     real_3 += len([i for i in y if i == 3])
    #     real_4 += len([i for i in y if i == 4])
    #     real_5 += len([i for i in y if i == 5])
    #
    # vis.bar(
    #     X = [tree_0_len, tree_1_len, tree_2_len,tree_3_len, tree_4_len, tree_5_len, guess_0, guess_1, guess_2,
    #          guess_3, guess_4, guess_5, real_0, real_1, real_2, real_3, real_4, real_5],
    #     opts= dict(title= f"Prediction Fails {epoch}",
    #            rownames=["tree_0", "tree_1", "tree_2", "tree_3", "tree_4", "tree_5", "guess_0", "guess_1", "guess_2",
    #                      "guess_3", "guess_4", "guess_5", "real_0", "real_1", "real_2", "real_3", "real_4", "real_5"]),
    #            win= f"bar3{epoch}"
    # )

    score /= sample_count
    # accuracy = correct / sample_count
    # print("\n", "\n")
    # print("Validation:")
    # print(f'Validation loss: {score}')
    # # print(f'Accuracy: {accuracy}')
    # print("\n", "\n")

    # vis.line(
    #     X = [epoch],
    #     Y = [accuracy],
    #     win= "mod_test11",
    #     name = "Dev Accuracy",
    #     update="append"
    #     )
    vis.line(
        X = [epoch],
        Y = [score],
        win= "mod_test11",
        name = "Dev Score",
        update="append"
        )

    epoch += 1
