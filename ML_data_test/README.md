# IQTREE ML

Run the script that runs IQTREE ML on your correct data representation:

  a) Labeled Data Representation:
  ~ Data: ([hot_encoded_seq_1, hot_encoded_seq_2, hot_encoded_seq_3, hot_encoded_seq_4], integer_label)

    1. In IQTREE_ML/ML_labeledData.py, change the parameter data_path to the path of your dev set
    2. Run IQTREE_ML/ML_labeledData.py and the IQTREE ML accuracy will be printed in the terminal

  b) Sequences dat file & Label txt file:

    1. In IQTREE_ML/ML_test.py, change the following parameters:

      a) INPUT_PATH - dat file containing your ordered sequences

      b) labels_path - txt file containing your ordered labels

      c) evomodel - suspected model of evolution

    2. Run IQTREE_ML/ML_test.py and the IQTREE ML accuracy will be printed in the terminal. Further, the file IQTREE_ML/ML_tree_output.dat will have your output

  c) IQTREE Evolution Model Selection:

    1. In IQTREE_ML/model_selection.py, change the following parameters:

      a) INPUT_PATH - path to your dev set

      b) MODEL_SET - set of evolution models to choose from

    2. Run IQTREE_ML/model_selection.py and the file IQTREE_ML/ML_model_output.dat will have your output
