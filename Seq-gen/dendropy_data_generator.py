import torch
import os
import numpy as np
import dendropy
import treeClassifier

data_size = {"train" : 0, #divisible by 15 = 3 tree_types * 5 tree_structures
             "dev" : 20,#3000,
             "test" : 0}
hot_encoding = {"A":[1, 0, 0, 0],
                "C":[0, 1, 0, 0],
                "T": [0, 0, 1, 0],
                "G": [0, 0, 0, 1]}

sequence_length = 20
scale = 1.0
evolution_model = "GTR"
equilibrium_base_frequencies =  "0.34,0.15,0.18,0.33"
reversable_rate_mx = "1.61, 3.82, 0.27, 1.56, 4.99, 1"

def _generate_tree(amount, tree_type, path):
    """
    Generates specified jukes-cantor trees into a path.dat

    amount - amount of datapoints in file
    tree_type - type of tree formed
    path - path to place data
    """
    #print(amount, tree_type, path)
    if evolution_model == "JC":
        os.system(f'seq-gen -m"HKY" -s{scale} -n{amount} -l{sequence_length} -t0.5 -fe <tree_types/{tree_type}.tre> {path}.dat')

    elif evolution_model == "GTR":
        os.system(f'seq-gen -m{evolution_model} -s{scale} -n{amount} -l{sequence_length} -f{equilibrium_base_frequencies} -r{reversable_rate_mx} <tree_types/{tree_type}.tre> {path}.dat')
    else:
        print("Error: Model not defined")

def _hot_encode(dna_seq):
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
        hot_encode_seq.append(hot_encoding[position])

    return hot_encode_seq


def generate_dendropy_data(simulator=dendropy.simulate.treesim.pure_kingman_tree, pop_size=1):
    """
    pop_size - size of the population from the which the coalescent process is sampled
    simulator - type of dendropy simulation
    """

    tns = dendropy.TaxonNamespace(["taxon1", "taxon2", "taxon3", "taxon4"], label="taxa")

    #loop over train, dev, test set
    for type_data, size in data_size.items():
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

def preprocess_dendropy_data():
    #get data from files
    print("reading data")
    train_data, dev_data, test_data = _read_all_files()

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
        print("get path")
        file_path = None
        for path, data in file_path_map.items():
            if type_data == data:
                file_path = path
                break

        assert file_path != None, "No file path - code error"

        #hot encode data
        print("hot encoding")
        data = []
        for sequences, label in type_data:
            new_datapoint = []
            for sequence in sequences:
                hot_encoded_seq = _hot_encode(sequence)
                new_datapoint.append(hot_encoded_seq)
            data.append((new_datapoint, label))

        #save data
        print("saving")
        save_data = np.array(data)
        np.save("data/" + file_path + ".npy", save_data)
        print("done saving")

def _read_all_files():
    """
    Reads all generated data
    """
    dataset = []

    for type_data in ["train", "dev", "test"]:
        seq_path = "data/" + type_data + "/" + type_data + "_dendropy.dat"
        label_path = "data/" + type_data + "/" + type_data + "_dendropy_labels.txt"
        data = _read_seq_gen_output(seq_path, label_path)

        if data == None:
            data = []

        dataset.append(data)

    return dataset

def _read_seq_gen_output(seq_path, label_path):
    """
    Reads data from a seq-gen output.
    Returns the sequences of the file and the appropriate labels too
    """
    try:
        file = open(seq_path, "r")
    except:
        return None

    #read in label data
    f = open(label_path, "r")
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

generate_dendropy_data()
preprocess_dendropy_data()
