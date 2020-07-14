print("imports")
import visdom
import pathlib
# import pickle
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
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = torch.nn.Linear(32, 3)

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



training_data = np.load("seq-gen/data/train/training_data.npy", allow_pickle = True)
validation_data = np.load("seq-gen/data/dev/development_data.npy", allow_pickle = True)
train_data = training_data.tolist()
print("loaded data")

#plotting
vis = visdom.Visdom()

#model Hyperparameters
model = _Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#weight initialization...
loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

BATCH_SIZE = 16
TRAIN_SIZE = 2000
epoch = 1

#Train
while epoch < 600:

    #TRAIN
    model.train()
    #randomly take 2000 datapoints
    epoch_train = random.sample(train_data, TRAIN_SIZE)
    sample_count, correct, score = 0, 0, 0.0

    for i in range(TRAIN_SIZE // BATCH_SIZE):
        data = epoch_train[i * BATCH_SIZE : (i+1) * BATCH_SIZE]

        x_list = []
        y_list = []

        for datapoint in data:
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
        opts= dict(title="ConvNet Tree Model 1",
               xlabel="Epochs",
               showlegend=True),
        win= "graph5",
        name = "Train Accuracy",
        update="append"
        )
    vis.line(
        X = [epoch],
        Y = [score],
        win= "graph5",
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
        win= "graph5",
        name = "Dev Accuracy",
        update="append"
        )
    vis.line(
        X = [epoch],
        Y = [score],
        win= "graph5",
        name = "Dev Score",
        update="append"
        )

    epoch += 1
