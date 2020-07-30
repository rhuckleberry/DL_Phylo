import os
import numpy as np

def dataset_merge(dataset_1, dataset_2, output_file):
    """
    Takes two datasets and merges them into the output_file
    ~removes input datasets too (if not commented out)

    Input:
    dataset_1 - path to first dataset
    dataset_2 - path to second dataset
    ouput_file - file name for final merged output dataset

    Output: file with merged data from both datasets
    """

    #load data
    data_1 = np.load(dataset_1, allow_pickle=True)
    data_2 = np.load(dataset_2, allow_pickle=True)

    #merge data
    merged_data = np.concatenate((data_1, data_2))

    #remove input datasets
    os.remove(dataset_1)
    os.remove(dataset_2)

    #save data
    np.save(output_file, merged_data)

    return merged_data

def new_sets(datasets, dev_percentage, output_path):
    """
    Takes datasets and then generates and dev and train set based on the given
    dev set percentage by merging the data in the given datasets and then
    dividing it up accordingly
    ~ calls standardize_data_types so that all types of data (labels) have the
      same amounts

    Input:
    datasets - list of strings of dataset paths
    dev_percentage - decimal of how much data should be in the dev set
    output_path - string to put final sets ("train" or "dev" will be appended
                  to end of string)
    """

    #load data
    load_data = []
    for dataset in datasets:
        data = np.load(dataset, allow_pickle=True)
        load_data.append(data)

    #concatenate data
    all_data = np.concatenate(load_data)

    #shuffle data
    np.random.shuffle(all_data)

    # _, all_data, lost_data = standardize_data_types(datasets)
    #
    # #shuffle data
    # np.random.shuffle(all_data)

    #divide all_data
    divide_index = int(len(all_data) * dev_percentage)
    new_dev = all_data[:divide_index]
    new_train = all_data[divide_index:]

    #print stats
    print("Dev size: ", len(new_dev))
    print("Train size: ", len(new_train))
    print("Total size: ", len(new_dev) + len(new_train))

    #save new datasets
    np.save(output_path + "_dev", new_dev)
    np.save(output_path + "_train", new_train)
    #np.save(output_path + "_lost", lost_data)

    return new_dev, new_train

def standardize_data_types(datasets):
    """
    Given labeled datasets it returns a maximum set of data that has same number
    of every type of label.
        ~Let A, B, C,... be sets of data. Returns set where every type of data
         has min(|A|, |B|, |C|, ...) datapoints

    Input:
    datasets - list of strings of dataset paths
    """

    #load data
    load_data = []
    for dataset in datasets:
        data = np.load(dataset, allow_pickle=True)
        load_data.append(data)

    #concatenate data
    all_data = np.concatenate(load_data)

    #shuffle data
    np.random.shuffle(all_data)

    #count data
    datatype_amount = {} #map of evolution model to (int of datapoints, list of datapoints)
    for sequences, label in all_data:
        datapoint = (sequences, label)
        if label in datatype_amount:
            datatype_amount[label][0] += 1
            datatype_amount[label][1].append(datapoint)
        else:
            datatype_amount[label] = [1, [datapoint]]



    #datapoint amount for each type of data
    amount = min([num for num, _ in datatype_amount.values()])

    #make set from datapoints
    new_dataset = []
    lost_datapoints = []
    for _, sequence_list in datatype_amount.values():
        new_dataset.extend(sequence_list[:amount])
        lost_datapoints.extend(sequence_list[amount:])

    #print stats
    print("Amount: ", amount)
    print("New Dataset Size: ", len(new_dataset))
    print("Lost Datapoints: ", len(lost_datapoints))

    return amount, new_dataset, lost_datapoints



if __name__ == "__main__":
    # dataset_1 = "ResNet_data/training_data__2.npy" #path to first dataset
    # dataset_2 = "ResNet_data/training_data__K80.npy" #path to second dataset
    # output_file = "ResNet_data/training_data__2.npy" #path to output of merged data
    # merged_data = dataset_merge(dataset_1, dataset_2, output_file)
    # print("Number of total datapoints: ", len(merged_data))


    # dataset_1 = "ResNet_data/JC_HKY_GTR/half_model_train.npy" #path to first dataset
    # dataset_2 = "ResNet_data/JC_HKY_GTR/half_model_dev.npy"
    # dataset_3 = "ResNet_data/JC_F81/F81_train.npy"
    # dataset_4 = "ResNet_data/JC_F81/F81_dev.npy"
    # dataset_5 = "ResNet_data/K80_train.npy"
    # dataset_6 = "ResNet_data/K80_train.npy"
    # dataset_7 = "ResNet_data/final_data.npy"
    # dataset_8 = "ResNet_data/combined_train.npy"
    # dataset_9 = "ResNet_data/combined_dev.npy"
    # dataset_10 = "ResNet_data/combined_lost.npy"
    # dataset_11 = "ResNet_data/whole_model_train.npy"
    # dataset_12 = "ResNet_data/whole_model_dev.npy"
    dataset_13 = "ResNet_data/test.npy"
    datasets = [dataset_13]

    dev_percentage = 0.3
    output_path = "ResNet_data/test8"
    _, _ = new_sets(datasets, dev_percentage, output_path)

    # _, _, _ = standardize_data_types(datasets)
