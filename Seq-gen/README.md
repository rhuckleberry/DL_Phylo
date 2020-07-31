# Seq-Gen

Choose the generator from below:

  a) Seq-gen generation

    1. In Seq-gen/data_generator.py, change the parameters at the top of the script

    2. Run Seq-gen/data_generator.py and your data will be output in the Seq-gen/data directory under the corresponding dev, test or train directory


  b) Dendropy generation - pure kingman, etc. (modify with simulator parameter)

    1. In Seq-gen/dendropy_data_generator.py, change the parameters at the top of the script. Also add the simulator parameter in the generate_dendropy_data function if you want to use a simulator that isn't the default pure kingman simulator

    2. Run Seq-gen/dendropy_data_generator.py and your data will be output in the Seq-gen/data directory under the corresponding dev, test or train directory


*You can ignore the permute_test directory as the scripts in it are no longer used (feel free to delete this directory)
*The tree_types folder holds the newick trees to generate the data and is used in generation scripts above
