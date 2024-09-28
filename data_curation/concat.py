import pandas as pd
import numpy as np
import os
from os.path import dirname
from multiprocessing import Pool
import time

# --------------- Initializing Paramaters ----------#

path = os.getenv("WORKING_DIR")
assert isinstance(path, str)

state_name = "CA"

path_stats = path + "/data/Road_Networks/" + state_name + "/"


# --------------- Functions ------------------------#

def read_csv(filename):
    return pd.read_csv(filename,low_memory=False)

def concat_files(path, final_file_name):
    """
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    """
    count = 0

    file_list = []
    for file_name in os.listdir(path):
        file_list.append(path + file_name)
    
    pool = Pool(processes=20)
    df_list = pool.map(read_csv,file_list)
    df = pd.concat(df_list)
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(final_file_name, index=False)


def concat_files_one_subfolder(path):
    """
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    """
    file_list =[]
    for folder in os.listdir(path):
        file_list.append(path + folder + "/edge_list.csv")
        
    pool = Pool(processes=20)
    df_list = pool.map(read_csv,file_list)
    
    edge_df = pd.concat(df_list)
    edge_df = edge_df.drop_duplicates().reset_index(drop=True)
    edge_df = edge_df.rename(columns={"u": "node_1", "v": "node_2"})
    
    file_list =[]
    for folder in os.listdir(path):
        file_list.append(path + folder + "/node_list.csv")
        
    pool = Pool(processes=20)
    df_list = pool.map(read_csv,file_list)
    node_df = pd.concat(df_list)
    node_df = node_df.drop_duplicates().reset_index(drop=True)
    node_df = node_df.rename(columns={"osmid": "node_id"})

    return node_df, edge_df


def concat_files_two_subfolder(path):
    """
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    """
    file_list = []
    for folder in os.listdir(path):
        for subfolder in os.listdir(path + folder + "/"):
            file_list.append(path + folder + "/" + subfolder + "/edge_list.csv")
     
    pool = Pool(processes=20)
    df_list = pool.map(read_csv,file_list)
    edge_df = pd.concat(df_list)
    edge_df = edge_df.drop_duplicates().reset_index(drop=True)
    edge_df = edge_df.rename(columns={"u": "node_1", "v": "node_2"})

    file_list = []
    for folder in os.listdir(path):
        for subfolder in os.listdir(path + folder + "/"):
            file_list.append(path + folder + "/" + subfolder + "/node_list.csv")
     
    pool = Pool(processes=20)
    df_list = pool.map(read_csv,file_list)
    node_df = pd.concat(df_list)
    node_df = node_df.drop_duplicates().reset_index(drop=True)
    node_df = node_df.rename(columns={"osmid": "node_id"})

    return node_df, edge_df


# ------------- Load Level-Wise Road Networks ---------------#
os.makedirs(path_stats + "Road_Network_Level/Nodes",exist_ok=True)
os.makedirs(path_stats + "Road_Network_Level/Edges",exist_ok=True)

start = time.time()
print("\nRoad Network of every Level:")

print("\tCities")
node_df, edge_df = concat_files_one_subfolder(
    path_stats
    + "Harvard Dataverse/"
    + state_name
    + "-cities-street_networks-node_edge_lists/"
)

node_df.to_csv(path_stats + "Road_Network_Level/Nodes/nodes_cities.csv", index=False)
edge_df.to_csv(path_stats + "Road_Network_Level/Edges/edges_cities.csv", index=False)
print(time.time()-start)
print("\tCounties")
node_df, edge_df = concat_files_one_subfolder(
    path_stats
    + "Harvard Dataverse/"
    + state_name
    + "-counties-street_networks-node_edge_lists/"
)

node_df.to_csv(path_stats + "Road_Network_Level/Nodes/nodes_counties.csv", index=False)
edge_df.to_csv(path_stats + "Road_Network_Level/Edges/edges_counties.csv", index=False)

print("\tNeighborhoods")
node_df, edge_df = concat_files_two_subfolder(
    path_stats
    + "Harvard Dataverse/"
    + state_name
    + "-neighborhoods-street_networks-node_edge_lists/"
)

node_df.to_csv(
    path_stats + "Road_Network_Level/Nodes/nodes_neighborhoods.csv", index=False
)
edge_df.to_csv(
    path_stats + "Road_Network_Level/Edges/edges_neighborhoods.csv", index=False
)


print("\tTracts")
node_df, edge_df = concat_files_one_subfolder(
    path_stats
    + "Harvard Dataverse/"
    + state_name
    + "-tracts-street_networks-node_edge_lists/"
)

node_df.to_csv(path_stats + "Road_Network_Level/Nodes/nodes_tracts.csv", index=False)
edge_df.to_csv(path_stats + "Road_Network_Level/Edges/edges_tracts.csv", index=False)


print("\tUrbanized Areas")
node_df, edge_df = concat_files_one_subfolder(
    path_stats
    + "Harvard Dataverse/"
    + state_name
    + "-urbanized_areas-street_networks-node_edge_lists/"
)

node_df.to_csv(
    path_stats + "Road_Network_Level/Nodes/nodes_urbanized_areas.csv", index=False
)
edge_df.to_csv(
    path_stats + "Road_Network_Level/Edges/edges_urbanized_areas.csv", index=False
)


# ----------- Concatenate the data --------------#

print("\nAppend all files")

print("\tNodes")
concat_files(
    path_stats + "Road_Network_Level/Nodes/",
    path_stats + "Road_Network_Nodes_" + state_name + ".csv",
)

print("\tEdges")
concat_files(
    path_stats + "Road_Network_Level/Edges/",
    path_stats + "Road_Network_Edges_" + state_name + ".csv",
)


print("\nRemove Duplicates and Save:")

print("\tNodes")
df_nodes = pd.read_csv(
    path_stats + "Road_Network_Nodes_" + state_name + ".csv", low_memory=False
)

df_nodes = df_nodes.drop_duplicates(["node_id"], keep="last")
df_nodes = df_nodes[["node_id", "x", "y"]]

df_nodes.to_csv(path_stats + "Road_Network_Nodes_" + state_name + ".csv", index=False)


print("\tEdges")
df_edges = pd.read_csv(
    path_stats + "Road_Network_Edges_" + state_name + ".csv", low_memory=False
)

df_edges = df_edges.drop_duplicates(["node_1", "node_2"], keep="last")
df_edges = df_edges[["node_1", "node_2", "oneway", "highway", "name", "length"]]

df_edges.to_csv(path_stats + "Road_Network_Edges_" + state_name + ".csv", index=False)


# ------------------------#
