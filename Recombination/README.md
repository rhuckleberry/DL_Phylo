# Recombination

Recombination Data Generation:

  1. Change recombination parameters in Recombination/data_generation/ctrlGenPar.py
  2. Change data generation parameters in Recombination/data_generation/main.py
    i) labels - each treetype must be run seperately right now
    ii) dataset sizes & proportions
    iii) sequence lengths, etc.
  3. Run Recombination/data_generation/main.py
  4. Run Recombination/data_generation/recombinationMerge.py making sure all data is in Recombination/data_generation/recombination_data
  5. If there are no lost datapoints, delete that file from Recombination/data_generation/recombination_data
  6. Move output datasets to Recombination/data_generation/test/data

Recombination Data Network/Testing

  a) Neural Network

  b) IQTREE ML Test
