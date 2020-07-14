from evomodels import *
import os

MODEL_MAP = {"JC" : 0,
             "K80" : 1,
             "F81" : 2,
             "HKY85" : 3,
             "F84" : 4,
             "GTR" : 5} #1-1 transformation from model to set of integers

MODEL_STRING_MAP = {JC : "JC",
                    K80 : "K80",
                    F81 : "F81",
                    HKY85 : "HKY85",
                    F84 : "F84",
                    GTR : "GTR"} #map functions to string names

hot_encoding = {"A":[1, 0, 0, 0],
                "C":[0, 1, 0, 0],
                "T": [0, 0, 1, 0],
                "G": [0, 0, 0, 1]}

data_amount = {JC : 0,
               K80 : 0,
               F81 : 10000,
               HKY85 : 10000,
               F84 : 10000,
               GTR : 10000} #map model to amount of data per model

sequence_length = 200
structure_file = "/Users/rhuck/Downloads/DL_Phylo/Seq-gen/tree_types/pure_kingman.tre"
sample_amount = 20 #datapoints per model instance

#generating data
def generate_data(data_amount, sequence_length, structure_file, sample_amount, intermediate_file = "model_data/intermediate.tre"):
    """
    Generate data from trees of structure_file structure and with the given
    evolution model
    """
    #open structure file
    structure_f = open(structure_file)
    structure_lines = structure_f.readlines()

    #generate data
    for evomodel, data_size in data_amount.items():
        #check data_size is allowed or skip if its 0
        if data_size == 0:
            continue
        assert data_size > 0

        f = open(intermediate_file, "x") #no carried over data
        f.close()

        model_instances = data_size // sample_amount
        print("Data Loss: ", data_size % sample_amount)

        for _ in range(model_instances):
            model, base_freq, t_ratio, rate_mx = evomodel()
            path = MODEL_STRING_MAP[evomodel] #string of model name
            intermediate_file = tree_structure_sample(sample_amount, structure_lines, intermediate_file)
            generate_tree(model, base_freq, t_ratio, rate_mx, 1, sequence_length, intermediate_file, path)
        os.remove(intermediate_file)

    #close structure file
    structure_f.close()

def tree_structure_sample(amount, structure_lines, intermediate_file = "model_data/intermediate.tre"):
    """
    Takes a file to a tree structure .tre file and samples amount tree structures
    from it and puts them in a different file to generate data from
    """

    #take amount size sample from all trees
    sampled_lines = random.sample(structure_lines, amount)

    #put sampled tree structure data into intermediate file
    intermediate_f = open(intermediate_file, "a")
    file_str = ""
    for line in sampled_lines:
        file_str += line
    intermediate_f.write(file_str)
    intermediate_f.close()
    #return path of used intermediate file
    return intermediate_file

def generate_tree(model, base_freq, t_ratio, rate_mx, amount, sequence_length, structure_file, path):
    """
    Generates tree with seq-gen into a path.dat file
    """
    if model == "GTR":
        os.system(f'seq-gen -m"GTR" -f{base_freq} -r{rate_mx} -n{amount} -l{sequence_length} \
                  <{structure_file}> model_data/{path}.dat')
    else:
        os.system(f'seq-gen -m"HKY" -f{base_freq} -t{t_ratio} -n{amount} -l{sequence_length} \
                  <{structure_file}> model_data/{path}.dat')

#preprocess data
def preprocess_data(output_path):
    all_data = []
    #loop over all files in directory
    directory = "model_data"
    for filename in os.listdir(directory):
        if filename.endswith(".dat"):
            label = MODEL_MAP[filename[:-4]]
            file_data = _read_file_data("model_data/" + filename, label)

            #hot encode data
            data = []
            for sequences, label in file_data:
                new_datapoint = []
                for sequence in sequences:
                    hot_encoded_seq = _hot_encode(sequence)
                    new_datapoint.append(hot_encoded_seq)
                data.append((new_datapoint, label))
            all_data.extend(data)

    #shuffle data
    random.shuffle(all_data)

    #save data
    save_data = np.array(all_data)
    np.save(output_path, save_data)

def _read_file_data(filename, label):
        """
        Reads data from file in given path

        Returns: file_data - list of tuples (sequences, label)
        """
        try:
            file = open(filename, "r")
        except:
            print("except")
            return None

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
        os.remove(filename)
        return file_data

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


generate_data(data_amount, sequence_length, structure_file, sample_amount)
output_path = "ResNet_data/final_data"
preprocess_data(output_path)
