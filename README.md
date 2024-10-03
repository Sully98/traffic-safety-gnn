# Traffic Safety GNN

The repository is for the study of the effectiveness of a GNN used to predict the safety of street intersections. 
This could then be extended to predict the safety of streets themselves. These are examples of node and edge 
regression tasks as we will be giving each a continuous safety score. This could then be overlayed with a real map
and could inform people on the safest routes to take. Using these scores, we could then study the safest roads
and intersections and see what features make them the safest. At first, the study will be limited to a small
region in California but could be generalized to any place where the original features could be recreated.

## Project Setup Guide

To run the notebooks in this repository, it is easiest to use conda and mamba. They are projects that make it easier to install python packages and have a way to run jupyter servers easily to use the jupyter notebooks.

#### Prerequisites
Downloaded the data from:
```
https://drive.google.com/drive/folders/1yWHl-eL9j63rNo4wBkXLGnCZb9L6JZ9V?usp=drive_link
```
The data is very large, so it may be smart to download the separate files inside then reconstruct the folder structure.

Make sure you have the following installed on your system:

- Conda
- Git

### Installation Steps

#### 1. Clone the GitHub Project

This will install the code in this repository to your local machine.

```
git clone https://github.com/Sully98/traffic-safety-gnn.git
```

#### 2. Navigate to the Project Directory

After you do this, you will be able to make the conda environment with the environment file.
```
cd traffic-safety-gnn
```
#### 3. Download and install Conda

- Download the latest version of Miniconda or Anaconda from the official website.
- Follow the installation instructions for your operating system.

#### 4. Make conda environment
Now that you are in the proper directory you can make a conda environment with all of the packages you need.
```
conda env create -f environment.yml
```

#### 5. Start the Jupyter Notebook Server
Launch the Jupyter Notebook server by running:
```
jupyter notebook
```
This command will open a new tab in your default web browser showing the Jupyter Notebook dashboard.

#### 6. Open the Notebook

In the Jupyter Notebook dashboard, navigate to the desired notebook file (with the .ipynb extension) and click on it to open.

#### 7. Make sure the all of the paths point to the directory you have stored the data in 

There will be sections in the code, namely instances where TrafficAccidentDataset is called, where specify a data path. Make sure this points to where you save the data.

#### Data Curation
The first step we need to do is to get the data into a format that will be readable by the GNN. This means 
actually creating a graph in the first place. The base graph will have just have the nodes and edges given an
index, then they will need to be connected. To enrich the graph, we can add features to the nodes and edges. 
The exact features given are subject to change based on how the GNN performs given certain features but some 
examples of features could be...
