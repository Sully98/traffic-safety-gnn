from data_loaders import TrafficAccidentDataset
# Creating the dataset as PyTorch Geometric dataset object
dataset = TrafficAccidentDataset(state_name = "CA", data_dir="../data/Final_Graphs",
                            #node_feature_type = args.node_feature_type,
                            use_static_edge_features=True,
                            #use_dynamic_node_features=args.load_dynamic_node_features,
                            #use_dynamic_edge_features=args.load_dynamic_edge_features,
                            #train_years=args.train_years,
                            num_negative_edges=100000000) 
# Loading the accident records and traffic network features of a particular month
data = dataset.load_monthly_data(year = 2016, month = 1)
# Pytorch Tensors storing the list of edges with accidents and accident numbers
accidents, accident_counts = data["accidents"], data["accident_counts"]
# Pytorch Tensors of node features, edge list, and edge features
x, edge_index, edge_attr = data["x"], data["edge_index"], data["edge_attr"]
print(data['data'])
