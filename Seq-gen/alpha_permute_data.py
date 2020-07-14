import torch
import os
import random
import numpy as np

seq_len = 200
data_size = {"train" : 100000,
             "dev" : 0,
             "test" : 0}
tree_type = "alpha" #only need to generate from one tree
tree_encoding = {"alpha" : 0,
                 "beta" : 1,
                 "gamma" : 2}
hot_encoding = {"A":[1, 0, 0, 0],
                "C":[0, 1, 0, 0],
                "T": [0, 0, 1, 0],
                "G": [0, 0, 0, 1]}

class Data_Generator:
    def __init__(self, seq_len, data_size, tree_type, tree_encoding, hot_encoding):
        self.seq_len = seq_len
        self.data_size = data_size
        self.tree_type = tree_type
        self.tree_encoding = tree_encoding
        self.hot_encoding = hot_encoding

    def generate_data(self):
        """
        Generates train/test/dev set data based on a uniform distribution
        """

        for type_data, amount in self.data_size.items():
            path = "data/" + type_data + "/" + self.tree_type
            self._generate_tree(amount, self.tree_type, path)
        pass

    def preprocess_data(self):
        """
        Preprocesses generated data into a new file where tree type generated
        data are merged, randomly sorted and hot-encoded
        """

        #get data from files
        train_data, dev_data, test_data = self._read_all_files()

        # #shuffle data -- tree types not next to each other
        # random.shuffle(train_data)
        # random.shuffle(dev_data)
        # random.shuffle(test_data)

        #hot encode data
        dataset = []
        for type_data in [train_data, dev_data, test_data]:
            data = []
            for sequences, label in type_data:
                new_datapoint = []
                for sequence in sequences:
                    hot_encoded_seq = self._hot_encode(sequence)
                    new_datapoint.append(hot_encoded_seq)
                data.append((new_datapoint, label))
            dataset.append(data)

        #seperating dataset variable and making np arrays
        training_data = np.array(dataset[0])
        development_data = np.array(dataset[1])
        testing_data = np.array(dataset[2])


        #save data
        np.save("data/train/training_data2.npy", training_data)
        np.save("data/dev/development_data2.npy", development_data)
        np.save("data/test/testing_data2.npy", testing_data)

        return training_data, development_data, testing_data

    def _read_all_files(self):
        """
        Reads all generated data
        """
        dataset = []
        for type_data in ["train", "dev", "test"]:
            path = "data/" + type_data + "/" + self.tree_type + ".dat"
            datapoint = self._read_file_data(path, self.tree_type)

            dataset.append(datapoint)

        return dataset


    def _read_file_data(self, path, tree):
        """
        Reads data from file in given path
        """
        file = open(path, "r")
        label = self.tree_encoding[tree]

        line_state = 0
        file_data = []
        datapoint = []
        for line in file:

            if line_state == 0:
                #reset datapoint after 4 lines
                datapoint = []

            elif line_state == 4 and len(datapoint) == 3:
                sequence = line[10:-1]
                datapoint.append(sequence)
                file_data.append((datapoint, label))

            else:
                sequence = line[10:-1]
                datapoint.append(sequence)

            line_state = (line_state + 1) % 5
        #os.remove(path)
        return file_data


    def _generate_tree(self, amount, tree_type, path):
        """
        Generates specified jukes-cantor trees into a path.dat

        amount - amount of datapoints in file
        tree_type - type of tree formed
        path - path to place data
        """
        print(amount, tree_type, path)
        os.system(f'seq-gen -m"HKY" -n{amount} -l{self.seq_len} -t0.5 <tree_types/{tree_type}.tre> {path}.dat')
        pass


    def _hot_encode(self, dna_seq):
        """
        Hot encodes a singular DNA sequence as follows:

        A --> [1, 0, 0, 0]
        C --> [0, 1, 0, 0]
        T --> [0, 0, 1, 0]
        G --> [0, 0, 0, 1]
        """

        #encoder = {"A":[1, 0, 0, 0],
                  # "C":[0, 1, 0, 0],
                  # "T": [0, 0, 1, 0],
                  # "G": [0, 0, 0, 1]}

        hot_encode_seq = []
        for position in dna_seq:
            hot_encode_seq.append(self.hot_encoding[position])

        return hot_encode_seq

jukes_cantor_gen = Data_Generator(seq_len, data_size, tree_type, tree_encoding, hot_encoding)
jukes_cantor_gen.generate_data()
training_data, development_data, testing_data = jukes_cantor_gen.preprocess_data()

#print(development_data)
