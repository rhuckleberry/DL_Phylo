#input an output of seq-gen
#return a ML file modeled on every set of quartet trees

import os
import treeCompare
import ML_Accuracy

INPUT_PATH = "/Users/rhuck/downloads/DL_Phylo/Seq-gen/data/dev/dev_dendropy.dat"
IQTREE_PATH = "/Users/rhuck/Downloads/DL_Phylo/ML_data_test/iqtree"
ML_PATH = "ml_tree_path" #directory name and write write files to
FINAL_PATH = "ML_tree_output.dat" #file name to final output file

labels_path = "/Users/rhuck/Downloads/DL_Phylo/Seq-gen/data/dev/dev_dendropy_labels.txt"
#path to labels to compare to in ML_accuracy to get accuracy

evomodel = "GTR"

def _read_dat_file_data(input_path):
    """
    Reads data from .dat file in given path
    -- makes a list of datapoints, where each datapoint has 4 .dat file lines
    """
    file = open(input_path, "r")

    line_state = 0
    file_data = []
    datapoint = []
    for line in file:

        if line_state == 4 and len(datapoint) == 4:
            datapoint.append(line)
            file_data.append(datapoint)
            datapoint = []

        else:
            datapoint.append(line)

        line_state = (line_state + 1) % 5
    # os.remove(path)
    # print(file_data)
    return file_data

def _run_ML(ml_path, iqtree_path, final_path, file_data, evomodel):
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
        write_f = open(WRITE_FILE_PATH, "a") #append mode

        #put quartet tree data in write_f file
        for line in datapoint:
            write_f.write(line)

        #run ML and delete not wanted files
        write_f.close()
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

def run_all_ML_test(input_path, iqtree_path, ml_path, final_path, evomodel):
    """
    Runs all the other functions and removed unnecessary files

    ~Final ML newick tree data is in FINAL_PATH
    """
    file_data = _read_dat_file_data(input_path)
    os.mkdir(ml_path) #creates a directory
    _run_ML(ml_path, iqtree_path, final_path, file_data, evomodel)

    #remove any other files in ml_path
    for filename in os.listdir(ml_path):
        os.remove(ml_path + "/" + filename)
    os.rmdir(ml_path) #removes directory


#Run program
run_all_ML_test(INPUT_PATH, IQTREE_PATH, ML_PATH, FINAL_PATH, evomodel)

accuracy, incorrect, total = ML_Accuracy.ML_accuracy(labels_path, FINAL_PATH)

# output_f = open(FINAL_PATH, "r")
# output = ML_Accuracy.read_labels_file(output_f)
# labels = [str(2) for i in range(len(output))]
# accuracy, incorrect, total = ML_Accuracy.accuracy(output, labels)
print("Number of Datapoints: ", total)
print("Incorrect: ", incorrect)
print("Accuracy: ", accuracy)
