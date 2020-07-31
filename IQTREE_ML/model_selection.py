#input an output of seq-gen
#return guesses of the model that generated the trees

import os
import numpy as np
import ML_Accuracy

INPUT_PATH = "/Users/rhuck/Downloads/DL_Phylo/Evolution_Model_Classification/ResNet_data/whole_model_dev.npy"
IQTREE_PATH = "iqtree-1.6.12-MacOSX/bin/iqtree"
ML_PATH = "ml_model_path" #directory name and write write files to
FINAL_PATH = "ML_model_output.dat" #file name to final output file

MODEL_SELECTION_TEST = "MF" #"TESTONLY"
MODEL_SET = "GTR"

def read_npy_file(input_path):
    """
    Reads npy file and turns it into a list of .dat file strings and a list of
    labels

    Outputs:
    file_data - list of .dat file strings
    labels - list of labels
    """
    data = np.load(input_path, allow_pickle=True)
    file_data = []
    labels = []

    sequence_number = len(data[0][0])
    sequence_length = len(data[0][0][0])

    for sequences, label in data:
        labels.append(label) #append label

        #build .dat file string
        file_string = str(sequence_number) + " " + str(sequence_length) + "\n"
        for index, sequence in enumerate(sequences, 1):
            line_string = "taxon" + str(index) + "    " + hot_unencode(sequence)
            file_string += line_string + "\n"

        file_data.append(file_string) #append .dat string

    return file_data, labels

def _run_selection(file_data, ml_path, iqtree_path, final_path, model_selection_test, model_set):
    """
    Runs ML on file -- puts quartet tree in a file then runs it, repeat...
    """

    #make file paths
    WRITE_FILE = "/removable_file.dat"
    WRITE_FILE_PATH = ml_path + WRITE_FILE
    write_f = open(WRITE_FILE_PATH, "x")

    final_f = open(final_path, "w")
    final_f = open(final_path, "a")

    #iterate through each quartet tree
    for datapoint in file_data:
        write_f = open(WRITE_FILE_PATH, "w") #removes old data in file

        #put quartet tree data in write_f file
        write_f.write(datapoint)

        #run ML and delete not wanted files
        write_f.close()
        os.system(iqtree_path + " -s " + WRITE_FILE_PATH + " -m " + \
                  model_selection_test + "-mset" + model_set)

        os.remove(WRITE_FILE_PATH + ".log")
        os.remove(WRITE_FILE_PATH + ".treefile")
        os.remove(WRITE_FILE_PATH + ".ckp.gz")
        os.remove(WRITE_FILE_PATH + ".model.gz")

        #access ML output data
        selection_file = open(WRITE_FILE_PATH + ".iqtree", "r")
        model_class = None
        for i, line in enumerate(selection_file, 1):
            if i == 29:
                line_list = line.split()
                model_class = line_list[-1] + "\n"
                break

            assert i <= 29

        assert model_class != None

        #put ML data in other file
        final_f.write(model_class)

        #remove ML file
        os.remove(WRITE_FILE_PATH + ".iqtree")

    #remove unnecessary directory and file
    os.remove(WRITE_FILE_PATH)

    #close final_f file, where output data is stored
    final_f.close()

def run_all_model_test(input_path, iqtree_path, ml_path, final_path, model_selection_test, model_set):
    """
    Runs all the other functions and removed unnecessary files

    ~Final ML newick tree data is in FINAL_PATH
    """
    file_data, labels = read_npy_file(input_path)
    os.mkdir(ml_path) #creates a directory
    _run_selection(file_data, ml_path, iqtree_path, final_path, model_selection_test, model_set)

    #remove any other files in ml_path
    for filename in os.listdir(ml_path):
        os.remove(ml_path + "/" + filename)
    os.rmdir(ml_path) #removes directory

    return labels


def hot_unencode(hot_encoded_seq):
    """
    Reverts a hot encoded sequence into a set of nucleotides again
    """
    hot_encoding = {(1, 0, 0, 0) : "A",
                    (0, 1, 0, 0) : "C",
                    (0, 0, 1, 0) : "T",
                    (0, 0, 0, 1) : "G"}

    dna_seq = ""
    for site in hot_encoded_seq:
        nucleotide = hot_encoding[tuple(site)]
        dna_seq += nucleotide

    return dna_seq

if __name__ == "__main__":

    labels = run_all_model_test(INPUT_PATH, IQTREE_PATH, ML_PATH, FINAL_PATH, MODEL_SELECTION_TEST, MODEL_SET)

    #make labels list a file
    #accuracy, incorrect, total = ML_Accuracy.ML_accuracy(FINAL_PATH, labels)
