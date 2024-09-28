
#--------------- Importing Libraries -------------#

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from os.path import dirname
from tqdm.contrib.concurrent import process_map
import swifter
import time
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

#--------------- Initializing Parameters ----------#

path = os.getenv("WORKING_DIR")
assert isinstance(path, str)

data_path = path + '/data'
#----------------- Functions -------------------#

def extract_nearest_street(edges_df,lat,lon):
    '''
    Extract the nodes of the nearest street given a latlong coordinate
    Methodology:
        Calculate the distance between 2 nodes
        Calculate the sum of distances between the point 
        and two nodes
        Extract the nodes/street with the minimum distance
    Parameters:
        edges_df (dataframe): details of the edges
        lat (float): latitude of the point
        lon (float): longitude of the point
    Returns:
        node 1, node 2
    '''
    #start = time.time()
    edges_df["street_dist_node_1"] = np.sqrt((lon - edges_df["node_1_x"])**2 + (lat - edges_df["node_1_y"])**2)
    edges_df["street_dist_node_2"] = np.sqrt((lon - edges_df["node_2_x"])**2 + (lat - edges_df["node_2_y"])**2)

    #edges_df["street_dist_node_1_plus_node_2"] = edges_df["street_dist_node_1"] + edges_df["street_dist_node_2"]

    edges_df["street_dist_diff"] = np.abs(edges_df["street_dist_node_1"] + edges_df["street_dist_node_2"] - edges_df["street_dist"])

    min_df = edges_df[edges_df["street_dist_diff"] == edges_df["street_dist_diff"].min()].reset_index(drop=True)
    #print(time.time() - start)
    return min_df.loc[0,"node_1"],min_df.loc[0,"node_2"]


def concat_files(path, final_file_name):
    '''
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    '''
    count = 0
    for file_name in tqdm(os.listdir(path)):
        try:
            df = pd.concat([df,pd.read_csv(path + file_name, low_memory=False)])
        except:
            df = pd.read_csv(path + file_name, low_memory=False)

        count += 1
    df = df.drop_duplicates().reset_index(drop=True)
    # df = df.rename(columns={"x":"lat","y":"lon","X":"lat","Y":"lon"})
    df.to_csv(final_file_name,index=False)


# Assuming extract_nearest_street is defined elsewhere
def process_row(args):
    i, row, edges_df = args
    node_1, node_2 = extract_nearest_street(edges_df, row["lat"], row["lon"])
    return node_1, node_2

def save_chunk(df_chunk, chunk_index, path, state_name):
    df_chunk.to_csv(f"{path}/Accidents/{state_name}/Accidents_Nearest_Street_{state_name}.csv", index=False)

def nearest_street_state(state_name, path):

    #---------------- Load Files ----------------#

    nodes_df = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Nodes_" + state_name + ".csv", low_memory=False)
    edges_df = pd.read_csv(path + "/Road_Networks/" + state_name + "/Road_Network_Edges_" + state_name + ".csv", low_memory=False)

    # Accident Data
    df = pd.read_pickle(path + "/Accidents/" + state_name + "/" + state_name + "_crash.pkl")

    #---------------- Clean Files ---------------#

    edges_df = edges_df[["node_1","node_2"]].drop_duplicates()

    nodes_df = nodes_df[["node_id","x","y"]].drop_duplicates()

    df["lat"] = pd.to_numeric(df["lat"])
    df["lon"] = pd.to_numeric(df["lon"])
    df["acc_count"] = df["acc_count"].astype(int)

    #------------------ Extract the nearest street details ------------------#

    # Merge the latlong coordinates of the nodes of all the edges
    edges_df = pd.merge(edges_df,nodes_df,left_on="node_1",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
    edges_df = edges_df.rename(columns={"x":"node_1_x","y":"node_1_y"})

    edges_df = pd.merge(edges_df,nodes_df,left_on="node_2",right_on="node_id",how="left").drop(columns=["node_id"],axis=1)
    edges_df = edges_df.rename(columns={"x":"node_2_x","y":"node_2_y"})

    # Calculate the distance between 2 nodes
    edges_df["street_dist"] = np.sqrt((edges_df["node_2_x"] - edges_df["node_1_x"])**2 + (edges_df["node_2_y"] - edges_df["node_1_y"])**2)

    # Extract the nodes of the nearest street from the given point
    print(f"Extract Nearest Street - {int(df.shape[0]/10000)} files:")
    start = time.time()
    #tqdm.pandas()

    df[['node_1','node_2']] = df.parallel_apply(lambda row: extract_nearest_street(edges_df, row["lat"], row["lon"]), axis=1, result_type='expand')
    df.to_csv(f"{path}/Accidents/{state_name}/Accidents_Nearest_Street_{state_name}.csv", index=False)
    end = time.time()
    print(end - start)
    
for state_name in ["CA"]:

    print(f"\n********* {state_name} ***********")
    os.makedirs(f"{data_path}/Accidents/{state_name}/Nearest_Street",exist_ok=True)
    
    nearest_street_state(state_name, data_path)

