# Training and Evaluating GNNs

We provide a script to train and evaluate graph neural networks on our dataset. To simplify the process of training and evaluating graph neural networks (GNNs) on our dataset for a specific state, we provide a Python script called `train.py`. You can customize the training process by specifying the following parameters inside the variable `args` in the file.:

- `--state_name` specifies the state of the dataset. Choose among `DE`, `IA`, `IL`, `MA`, `MD`, `MN`, `MT`, `NV`. 
- `--encoder` specifies the model to encode node features. `none` indicate only using MLP as the predictor. We provide the following encoders: `gcn`, `graphsage`, `gin`, and `dcrnn`. 
  -  `--num_gnn_layers` specifies the number of encoder layers.
  - `--hidden_channels` specifies the hidden model width. 
- `--train_years`, `--valid_years`, and `--test_years` specifies the splitting of datasets for training, validation, and testing. 
- it has been found that num_negative_edges at 10000 has worked the best.

Here is an example of bash script for training a GCN model on MA dataset below:

```bash
python train.py 
```

This will then save a series of models with the name epoch_n.pth. These models will then be loaded by the model in the notebook to validate the performance.