# DL_Phylo
Deep Learning for Phylogenetic Inference

Set-Up:
1. Add the directory Evolution_Model_Classification/ResNet_data
2. Add the directories:
    a)  Seq-gen/data/train
    b) Seq-gen/data/dev
    c) Seq-gen/data/test
    
Data Generation: 
~Change the parameters at the top of the script to generate the data you wish to specify
1. Quartet Tree Classification Data: choose one of the following
    a) Seq-gen/dendropy_data_generator.py -- for pure kingman structured data
    b) Seq-gen/data_generator.py -- for other, not pure kingman structued data (uniform distribution of structures)
2. Evolution Model Classification Data
    a) Evolution_Model_Classification/model_data_generator.py
    
Data Set Merging:
~Use this to merge datasets that you have already generated to save time
1. Evolution_Model_Classification/dataset_merge.py
    
Run Models:
1. Quartet Tree Classification Data
    a) Deep Learning Models:
    
        i) phylo_ResNet.py -- ResNet
        
        ii) phylo_ConvNet.py -- ConvNet
        iii) inception_net.py -- Inception Network
    b) Maximum Likelihood
        i) ML_data_test/ML_test.py
    c) Evolution Model Classification Data
        i) Evolution_Model_Classification/test_network.py -- Inception Network
        ii) Evolution_Model_Classification/evomodel_ResNet.py -- ResNet
        
*Files/functions about "permuting" or "transforming" can be largely ignored. They are vestiges of our old data generation/augementation. I leave the scripts here for now in case they later prove useful

*treeClassifer.py and treeCompare.py are practically the same scripts, but treeClassifier.py is the updated version of treeCompare.py and can effectively replace it

    
    

