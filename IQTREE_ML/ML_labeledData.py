import numpy as np
import datetime
import os
import treeCompare
import ML_Accuracy

def _tuple_to_dat(quartet_tree):
    """
    Takes a tuple quartet tree representation and outputs a .dat file string representation

    Input:
    - quartet_tree (tuple representation)
    - output_file (path to output.dat file)

    Output: .dat file string representation

    *Assume quartet tree (4 leaves)
    """

    #form .dat file string
    sequences, label = quartet_tree
    sequence_length = len(sequences[0])
    dat_str = f" 4  {sequence_length}"
    for integer, sequence in enumerate(sequences):
        nucleotide_sequence = _unhotencode(sequence)
        dat_str += f"\ntaxon{integer + 1}    {nucleotide_sequence}"
    return dat_str

def _unhotencode(sequence):
    """
    Given a hot envoded sequence, it unhotencodes it
    """
    un_hotencoding = {(1, 0, 0, 0) : "A",
                    (0, 1, 0, 0) : "C",
                    (0, 0, 1, 0) : "T",
                    (0, 0, 0, 1) : "G"}

    new_sequence = ""
    for hot_encoding in sequence:
        new_sequence += un_hotencoding[tuple(hot_encoding)]

    return new_sequence

def _run_ML(file_data, ml_path="ml_tree_path", iqtree_path="/Users/rhuck/Downloads/DL_Phylo/ML_data_test/iqtree",
            final_path="ML_output.dat", evomodel="GTR"):
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
    for dat_str in file_data:
        write_f = open(WRITE_FILE_PATH, "w") #removes old data in file
        write_f.write(dat_str)
        write_f.close()

        #run ML and delete not wanted files
        print(iqtree_path + " -s " + WRITE_FILE_PATH + " -m " + evomodel)
        os.system(iqtree_path + " -s " + WRITE_FILE_PATH + " -m " + evomodel)

        os.remove(WRITE_FILE_PATH + ".mldist")
        os.remove(WRITE_FILE_PATH + ".log")
        os.remove(WRITE_FILE_PATH + ".iqtree")
        os.remove(WRITE_FILE_PATH + ".ckp.gz")
        os.remove(WRITE_FILE_PATH + ".bionj")

        #access ML output data
        ML_data = open(WRITE_FILE_PATH + ".treefile", "r")
        newick_tree = []
        for line in ML_data:
            assert(newick_tree == [])
            treeClass = treeCompare.getClass(line)
            newick_tree = str(treeClass) + " " + line

        #put ML data in other file
        final_f.write(newick_tree)

        #remove ML file
        os.remove(WRITE_FILE_PATH + ".treefile")

    #remove unnecessary directory and file
    os.remove(WRITE_FILE_PATH)

    #close final_f file, where output data is stored
    final_f.close()

def ML_test(tuple_data_path):
    """
    Runs all of the ML tests given tuple representation tree topology data
    """
    #load data
    print("[LOG] Loading in data")
    tuple_data = np.load(tuple_data_path, allow_pickle=True)

    #get .dat file representation
    print("[LOG] Forming .dat file representation")
    file_data = []
    for quartet_tree in tuple_data:
        dat_str = _tuple_to_dat(quartet_tree)
        file_data.append(dat_str)

    #run ML
    print("[LOG] Running ML")
    ml_path = "ml_tree_path"
    os.mkdir(ml_path) #creates a directory
    _run_ML(file_data) # run program
    for filename in os.listdir(ml_path): #remove any other files in ml_path
        os.remove(ml_path + "/" + filename)
    os.rmdir(ml_path) #removes directory

    #calculate accuracy
    print("[LOG] Calculating Accuracy")
    accuracy = ML_Accuracy.ML_accuracy(tuple_data_path, "ML_output.dat")

    return accuracy

if __name__ == "__main__":
    begin_time = datetime.datetime.now()

    tuple_data_path = "/Users/rhuck/Downloads/DL_Phylo/Recombination/test_data/test1_fact5_sl10000_dev.npy"
    accuracy, incorrect, total = ML_test(tuple_data_path)
    print("Accuracy: ", accuracy)
    print("Incorrect: ", incorrect)
    print("Total: ", total)

    print("Execution Time: ", datetime.datetime.now() - begin_time)
