#Run this program after ML_Test to get the accuracy of ML when compared to
#other labels

ML_OUTPUT_PATH = "ML_output.dat"
labels_path = "/Users/rhuck/Downloads/DL_Phylo/Seq-gen/data/dev/dev_dendropy_labels.txt"



def ML_accuracy(ml_output_path, labels_path):
    """
    Gets output and labels from path and then finds ML accuracy

    Input:
    ml_output_path - path to the output of the output guesses from ML
    labels_path - path to the true labels
    file_type - string of the type of file to get labels from ("label" or "npy")
    """

    #choose correct file reading function
    read_file_func = None
    if labels_path[-3:] == "txt":
        read_file_func = read_labels_file
    elif labels_path[-3:] == "npy":
        read_file_func = read_npy_file
    else:
        print("Error: File type not recognized")
    assert read_file_func != None

    #read files
    output_f = open(ml_output_path, "r")
    labels_f = open(labels_path, "r")
    output = read_file_func(output_f)
    labels = read_file_func(labels_f)

    #return ML accuracy
    return accuracy(output, labels)

def read_labels_file(file):
    """
    Takes a label file or ML output .dat file and returns an array of its
    guesses/labels
    """
    labels = []

    for line in file:
        labels.append(line[0])

    return labels

def read_npy_file(file):
    """
    Takes an .npy file and returns an array of its output/labels
    """
    data = np.load(file, allow_pickle=True)
    labels = []

    for sequences, label in data:
        labels.append(label)

    return labels


def accuracy(output, labels):
    """
    output - list of output of ML or any function
    labels - list of labels to compare the output to
    """

    assert len(output) == len(labels), "output, labels not same length"

    correct = 0
    incorrect = {}

    for index, datapoint in enumerate(output):
        if datapoint == labels[index]:
            correct += 1
        else:
            incorrect_datapoint = (datapoint, labels[index])
            if incorrect_datapoint in incorrect:
                incorrect[incorrect_datapoint] += 1
            else:
                incorrect[incorrect_datapoint] = 1

    total = len(labels)

    return correct / total, incorrect, total

if __name__ == "__main__":
    accuracy, incorrect = ML_accuracy(ML_OUTPUT_PATH, labels_path)
    print(accuracy)
