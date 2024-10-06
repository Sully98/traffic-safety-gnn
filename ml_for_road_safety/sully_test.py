class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

from models import LinkPredictor,GNN
from evaluators import Evaluator
import torch
import itertools
from data_loaders import TrafficAccidentDataset
import os
import numpy as np

args = dict(batch_size=32768, device=0, dropout=0.0, encoder='gcn', epochs=4, eval_steps=2, hidden_channels=256, input_channels=128, input_time_steps=4, jk_type='last', load_dynamic_edge_features=True, load_dynamic_node_features=True, load_model_dir='none', load_model_name='epoch_50.pth', load_static_edge_features=True, log_steps=1, lr=0.01, node_feature_name='MA_ppr_128.npy', node_feature_type='verse', num_gnn_layers=2, num_negative_edges=100000, num_predictor_layers=2, runs=1, sam_rho=0.05, sample_batch_size=10000, sample_node=False, sp_lambda=0.0001, state_name='CA', supcon_lam=0.9, supcon_tmp=0.3, test_years=[2018], train_accident_regression=False, train_sam=False, train_soft_penalty=False, train_supcon=False, train_volume_regression=False, train_years=[2016], use_time_series=False, valid_years=[2017])
args = dotdict(args)
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = TrafficAccidentDataset(state_name = 'CA', data_dir="/users/sbillingsley/traffic-safety-gnn/data/Final_Graphs",
                            node_feature_type = args.node_feature_type,
                            use_static_edge_features=args.load_static_edge_features,
                            use_dynamic_node_features=args.load_dynamic_node_features,
                            use_dynamic_edge_features=args.load_dynamic_edge_features,
                            train_years=args.train_years,
                            num_negative_edges=args.num_negative_edges) 

from trainers import *
task_type = "classification"
evaluator = Evaluator(type=task_type)

data = dataset.data
in_channels_node = data.x.shape[1] if data.x is not None else 0
in_channels_node = (in_channels_node + 6) if args.load_dynamic_node_features else in_channels_node

in_channels_edge = data.edge_attr.shape[1] if args.load_static_edge_features else 0
in_channels_edge = in_channels_edge + 1 if args.load_dynamic_edge_features else in_channels_edge
print(in_channels_edge)
in_channels_edge = 53
model = GNN(in_channels_node, in_channels_edge, hidden_channels=args.hidden_channels, 
                    num_layers=args.num_gnn_layers, dropout=args.dropout,
                    JK = args.jk_type, gnn_type = args.encoder, num_nodes=data.num_nodes).to(device)


feature_channels = in_channels_node if args.encoder == "none" else args.hidden_channels
if_regression = args.train_accident_regression or args.train_volume_regression
predictor = LinkPredictor(in_channels=feature_channels*2 + in_channels_edge, 
                            hidden_channels=args.hidden_channels, 
                            out_channels=1,
                            num_layers = args.num_predictor_layers,
                            dropout=args.dropout,
                            if_regression=if_regression).to(device)
params = itertools.chain(model.parameters(), predictor.parameters())
optimizer = torch.optim.Adam(params, lr=args.lr)

trainer = Trainer(model, predictor, dataset, optimizer, evaluator,
                            train_years = args.train_years,
                            valid_years = args.valid_years,
                            test_years = args.test_years,
                            epochs=args.epochs,
                            batch_size = args.batch_size,
                            eval_steps=args.eval_steps,
                            device = device,
                            log_metrics=['ROC-AUC', 'F1', 'AP', 'Recall', 'Precision'],
                            use_time_series=args.use_time_series, input_time_steps=args.input_time_steps)


results = {}
for run in range(args.runs):
    predictor.reset_parameters()
    params = itertools.chain(model.parameters(), predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    log = trainer.train(num_months=12)

    for key in log.keys():
        if key not in results:
            results[key] = []
        results[key].append(log[key])
for key in results.keys():
    print("{} : {:.2f} +/- {:.2f}".format(key, np.mean(results[key]), np.std(results[key])))