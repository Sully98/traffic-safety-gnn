# Traffic Safety GNN

The repository is for the study of the effectiveness of a GNN used to predict the safety of street intersections. 
This could then be extended to predict the safety of streets themselves. These are examples of node and edge 
regression tasks as we will be giving each a continuous safety score. This could then be overlayed with a real map
and could inform people on the safest routes to take. Using these scores, we could then study the safest roads
and intersections and see what features make them the safest. At first, the study will be limited to a small
region in California but could be generalized to any place where the original features could be recreated.

#### Data Curation
The first step we need to do is to get the data into a format that will be readable by the GNN. This means 
actually creating a graph in the first place. The base graph will have just have the nodes and edges given an
index, then they will need to be connected. To enrich the graph, we can add features to the nodes and edges. 
The exact features given are subject to change based on how the GNN performs given certain features but some 
examples of features could be...
