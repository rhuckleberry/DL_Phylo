import torch
import os
import random
import numpy as np
import dendropy
import treeClassifier

tree_struct = {"alpha" : ["((taxon1:_, taxon2:_):_, (taxon3:_, taxon4:_):_);",
                          "(((taxon1:_, taxon2:_):_, taxon3:_):_, taxon4:_);",
                          "(((taxon1:_, taxon2:_):_, taxon4:_):_, taxon3:_);",
                          "(((taxon3:_, taxon4:_):_, taxon1:_):_, taxon2:_);",
                          "(((taxon3:_, taxon4:_):_, taxon2:_):_, taxon1:_);"],
                "beta" : ["((taxon1:_, taxon3:_):_, (taxon2:_, taxon4:_):_);",
                          "(((taxon1:_, taxon3:_):_, taxon2:_):_, taxon4:_);",
                          "(((taxon1:_, taxon3:_):_, taxon4:_):_, taxon2:_);",
                          "(((taxon2:_, taxon4:_):_, taxon1:_):_, taxon3:_);",
                          "(((taxon2:_, taxon4:_):_, taxon3:_):_, taxon1:_);"],
                "gamma" : ["((taxon1:_, taxon4:_):_, (taxon2:_, taxon3:_):_);",
                           "(((taxon1:_, taxon4:_):_, taxon2:_):_, taxon3:_);",
                           "(((taxon1:_, taxon4:_):_, taxon3:_):_, taxon2:_);",
                           "(((taxon2:_, taxon3:_):_, taxon1:_):_, taxon4:_);",
                           "(((taxon2:_, taxon3:_):_, taxon4:_):_, taxon1:_);"]}

data_size = {"train" : 0,#54000 *2, #divisible by 15 = 3 tree_types * 5 tree_structures
             "dev" : 10000,
             "test" : 0}
tree_types = ["alpha", "beta", "gamma"]
tree_encoding = {"alpha" : 0,
                 "beta" : 1,
                 "gamma" : 2}
hot_encoding = {"A":[1, 0, 0, 0],
                "C":[0, 1, 0, 0],
                "T": [0, 0, 1, 0],
                "G": [0, 0, 0, 1]}

sequence_length = 200
evolution_model = "GTR"
equilibrium_base_frequencies =  "0.34,0.15,0.18,0.33"
reversable_rate_mx = "1.61, 3.82, 0.27, 1.56, 4.99, 1"
# equilibrium_base_frequencies = "0.2112,0.2888,0.2896,0.2104"
# reversable_rate_mx = "0.2173,0.9798,0.2575,0.1038,1,0.2070"

alpha = (1, 1, 1, 1) #dirichlet distribution parameter


def generate_all_structures(amount, tree_type, path):
    #must divide out the tree stuctures in the tree.tre files
    num_tree_structures = 5 ##change to number of structures in tree.tre file

    new_amount = amount // num_tree_structures
    extra_amount = amount % num_tree_structures

    #if not divideable by num_tree_structures, will add num_tree_structures-extra_amount datapoints
    if extra_amount != 0:
        new_amount += 1
        print("Extra Added Data: " + str(num_tree_structures - extra_amount))
    print("here", tree_type)

    _generate_tree(new_amount, tree_type, path)

def generate_random_branch_length(amount, tree_type, path):
    """
    Generates specified jukes-cantor trees into a path.dat

    amount - amount of datapoints in file
    tree_type - type of tree formed
    path - path to place data
    """
    assert amount > 0

    _randomize_branch_lengths(amount, tree_type)
    _generate_tree(1, "random", path)


def _generate_tree(amount, tree_type, path):
    """
    Generates specified jukes-cantor trees into a path.dat

    amount - amount of datapoints in file
    tree_type - type of tree formed
    path - path to place data
    """
    #print(amount, tree_type, path)
    if evolution_model == "JC":
        os.system(f'seq-gen -m"HKY" -n{amount} -l{sequence_length} -t0.5 -fe <tree_types/{tree_type}.tre> {path}.dat')

    elif evolution_model == "GTR":
        os.system(f'seq-gen -m{evolution_model} -n{amount} -l{sequence_length} -f{equilibrium_base_frequencies} -r{reversable_rate_mx} <tree_types/{tree_type}.tre> {path}.dat')
    else:
        print("Error: Model not defined")

def _seq_gen_label(sequence_list):
    """
    sequence_list - list of seq-gen relations (sets) from output file
        ~Ex: ["taxon1", "taxon2", "taxon3", "taxon4"] -- label 0
    """
    label_map = {0 : [set(["taxon1", "taxon2"]), set(["taxon3", "taxon4"])],
                 1 : [set(["taxon1", "taxon3"]), set(["taxon2", "taxon4"])],
                 2 : [set(["taxon1", "taxon4"]), set(["taxon2", "taxon3"])]}

    #find related pair
    pair = set([sequence_list[0], sequence_list[1]])

    #find label
    for label, partition in label_map.items():
        if pair in partition:
            return label

    print("Error: matches none of the labels!  ", sequence_list)

def _randomize_branch_lengths(amount, tree_type):
    """
    Writes into random.tre file a new tree_type tree with randomly generated
    branch lengths
    """

    f = open("tree_types/random.tre", "w")
    f = open("tree_types/random.tre", "a")

    amount_counter = 0
    while amount_counter < amount:
        #loop over all tree structures
        for tree_structure in tree_struct[tree_type]:

            #protect from adding too many trees
            if amount_counter < amount:
                amount_counter += 1

                rand_tree_struct = tree_structure
                #loop over all randomly chosen branch lengths
                for _ in range(6):
                    branch_len = str(max(random.gauss(0.2, 0.2), 0.01))
                    rand_tree_struct = rand_tree_struct.replace("_", branch_len, 1)

                f.write(rand_tree_struct + "\n")

            else:
                break #exits for loop and will also end while loop

    assert amount_counter == amount, "Not correct number of trees"

    f.close()

def _dirichlet_sample():
    """
    Returns a single vector sampled from dirichlet(alpha)
    ~indices in a vector sampled from a dirichlet sum to 1 -- good for probabilities

    -Using distribution to get values for equilibrium_base_frequencies
    """

    sample = np.random.dirichlet(alpha) #can add size parameter to get more samples
    return sample


class Data_Generator:
    def __init__(self, sequence_length, data_size, tree_types, tree_encoding, hot_encoding, tree_struct):
        self.sequence_length = sequence_length
        self.data_size = data_size
        self.tree_types = tree_types
        self.tree_encoding = tree_encoding
        self.hot_encoding = hot_encoding
        self.tree_struct = tree_struct

    def generate_dendropy_data(self, simulator=dendropy.simulate.treesim.pure_kingman_tree, pop_size=1):
        """
        pop_size - size of the population from the which the coalescent process is sampled
        simulator - type of dendropy simulation
        """

        tns = dendropy.TaxonNamespace(["taxon1", "taxon2", "taxon3", "taxon4"], label="taxa")

        #loop over train, dev, test set
        for type_data, size in self.data_size.items():
            trees = []
            labels = []

            if size == 0:
                continue
            assert size > 0

            #generate tree structures
            for _ in range(size):
                tree = str(simulator(tns, pop_size=1))
                #tree label
                label = treeClassifier.getClass(tree)
                labels.append(label)
                trees.append(tree)

            #combine into strings
            label_str = ""
            tree_str = ""

            for index, tree in enumerate(trees):
                label_str += str(labels[index]) + "\n"
                tree_str += tree + ";" + "\n"

            # #make labels file
            label_f = open("data/"+type_data+"/"+type_data+"_dendropy_labels.txt", "w")
            label_f.write(label_str)
            label_f.close()

            #make .dat file for trees
            tree_f = open("tree_types/dendropy.tre", "w")
            tree_f.write(tree_str)
            tree_f.close()

            #seq-gen generate tree structures
            print(type_data)
            path = "data/" + type_data + "/" + type_data + "_dendropy"
            _generate_tree(1, "dendropy", path)

    def preprocess_dendropy_data(self):
        #get data from files
        train_data, dev_data, test_data = self._read_all_files(True)

        #mapping of data to their corresponding file paths
        file_path_map = {"train/training_data" : train_data,
                         "dev/development_data" : dev_data,
                         "test/testing_data" : test_data}

        #loop over train, dev, test set data
        for type_data in [train_data, dev_data, test_data]:
            if len(type_data) == 0:
                print("file is empty!")
                continue

            #get file path
            file_path = None
            for path, data in file_path_map.items():
                if type_data == data:
                    file_path = path
                    break

            assert file_path != None, "No file path - code error"

            #hot encode data
            data = []
            for sequences, label in type_data:
                new_datapoint = []
                for sequence in sequences:
                    hot_encoded_seq = self._hot_encode(sequence)
                    new_datapoint.append(hot_encoded_seq)
                data.append((new_datapoint, label))

            #save data
            save_data = np.array(data)
            np.save("data/" + file_path + ".npy", save_data)


    def generate_data(self, _generation_model):
        """
        Generates train/test/dev set data based on a uniform distribution

        _generation_model - function of model of tree generation
        """

        #loop over train, dev, test set
        for type_data, size in self.data_size.items():
            num_tree_types = len(self.tree_types) #generally 3: alpha, beta, gamma
            amount = size // num_tree_types
            extra_amount = size % num_tree_types

            if amount == 0:
                continue

            assert amount > 0

            #loop over tree types
            for i, tree in enumerate(self.tree_types):
                path = "data/" + type_data + "/" + tree

                if i == 0:
                    _generation_model(amount+extra_amount, tree, path)
                else:
                    _generation_model(amount, tree, path)

    def preprocess_data(self):
        """
        Preprocesses generated data into a new file where tree type generated
        data are merged, randomly sorted and hot-encoded
        """

        #get data from files
        train_data, dev_data, test_data = self._read_all_files()

        #shuffle data -- tree types not next to each other
        random.shuffle(train_data)
        random.shuffle(dev_data)
        random.shuffle(test_data)

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
        np.save("data/train/training_data.npy", training_data)
        np.save("data/dev/development_data.npy", development_data)
        np.save("data/test/testing_data.npy", testing_data)

        return training_data, development_data, testing_data

    def _read_all_files(self, isDendropy=False):
        """
        Reads all generated data
        """
        dataset = []

        if isDendropy:
            for type_data in ["train", "dev", "test"]:
                path = "data/" + type_data + "/" + type_data + "_dendropy.dat"
                data = self._read_seq_gen_output(path)

                if data == None:
                    data = []

                dataset.append(data)

        else:
            for type_data in ["train", "dev", "test"]:
                data = []
                for tree in self.tree_types:
                    path = "data/" + type_data + "/" + tree + ".dat"
                    datapoint = self._read_file_data(path, tree)

                    if datapoint == None:
                        continue

                    data += datapoint

                dataset.append(data)

        return dataset

    def _read_seq_gen_output(self, path):
        """
        Reads data from a seq-gen output.
        Returns the sequences of the file and the appropriate labels too
        """
        try:
            file = open(path, "r")
        except:
            return None

        #read in label data
        f = open("data/train/train_dendropy_labels.txt", "r")
        labels = []
        for line in f:
            labels.append(int(line[0]))

        line_state = 0
        file_data = []
        datapoint_index = 0
        # datapoint = [None, None, None, None] #assumes is quartet tree
        # ouput_order = []
        for line in file:

            if line_state == 0:
                #reset datapoint after 4 lines
                datapoint = [None, None, None, None] #assumes is quartet tree
                output_order = []

            elif line_state == 4:
                list_index = int(line[5]) - 1
                sequence = line[10:-1]
                taxon_name = line[:6]

                output_order.append(taxon_name)

                assert datapoint[list_index] == None
                datapoint[list_index] = sequence

                for i in line:
                    assert i != None

                label = labels[datapoint_index]
                file_data.append((datapoint, label))

                datapoint_index += 1

            else:
                list_index = int(line[5]) - 1
                sequence = line[10:-1]
                taxon_name = line[:6]

                output_order.append(taxon_name)

                assert datapoint[list_index] == None
                datapoint[list_index] = sequence

            line_state = (line_state + 1) % 5
        #os.remove(path)
        return file_data


    def _read_file_data(self, path, tree):
        """
        Reads data from file in given path
        """
        try:
            file = open(path, "r")
        except:
            return None

        label = self.tree_encoding[tree]

        line_state = 0
        file_data = []
        datapoint = [None, None, None, None] #assumes is quartet tree
        for line in file:

            if line_state == 0:
                #reset datapoint after 4 lines
                datapoint = [None, None, None, None]

            elif line_state == 4:
                list_index = int(line[5]) - 1
                sequence = line[10:-1]

                assert datapoint[list_index] == None
                datapoint[list_index] = sequence

                for i in line:
                    assert i != None

                file_data.append((datapoint, label))

            else:
                list_index = int(line[5]) - 1
                sequence = line[10:-1]

                assert datapoint[list_index] == None
                datapoint[list_index] = sequence

            line_state = (line_state + 1) % 5
        #os.remove(path)
        return file_data

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

gtr_gen = Data_Generator(sequence_length, data_size, tree_types, tree_encoding, hot_encoding, tree_struct)
# gtr_gen.generate_dendropy_data()
# gtr_gen.preprocess_dendropy_data()


gtr_gen.generate_data(generate_random_branch_length)
#training_data, development_data, testing_data = gtr_gen.preprocess_data()
