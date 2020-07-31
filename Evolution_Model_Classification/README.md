# Evolution Model Classification

Data Generation:

  ~ To change parameters that generate the prior distribution of evolution models, go to Evolution_Model_Classification/data_generation/evomodels.py

  1. In Evolution_Model_Classification/data_generation/model_data_generator.py, change the parameters at the top

  2. Run Evolution_Model_Classification/data_generation/model_data_generator.py and Evolution_Model_Classification/data_generation/ResNet_data will hold your output data

Dataset Merge:

  ~ This will also form your train/dev datasets

  1. In Evolution_Model_Classification/data_generation/dataset_merge.py, form a list of the paths to the data that you wish to concatenate and your output path

  2. Run Evolution_Model_Classification/data_generation/dataset_merge.py your specified output path will hold the merged datasets

Run Classification Networks:

  a) Inception Network

    1. In Evolution_Model_Classification/evolution_networks/inception_network.py, add the paths to your test and dev set

    2. Run "Visdom" in terminal

    3. Run Evolution_Model_Classification/evolution_networks/inception_network.py

  b) Residual Neural Network

      1. In Evolution_Model_Classification/evolution_networks/evomodel_ResNet.py, add the paths to your test and dev set

      2. Run "Visdom" in terminal

      3. Run Evolution_Model_Classification/evolution_networks/evomodel_ResNet.py
