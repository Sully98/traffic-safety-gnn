from pyDataverse.api import NativeApi, DataAccessApi
from pyDataverse.models import Dataverse
import argparse
import os
import zipfile

path = os.getenv("WORKING_DIR")
assert isinstance(path, str)

data_dir = path + "/data"
raw_dir = data_dir + "/raw"
road_networks = data_dir + "/Road_Networks"
os.makedirs(data_dir,exist_ok=True)
os.makedirs(raw_dir,exist_ok=True)
os.makedirs(road_networks,exist_ok=True)
parser = argparse.ArgumentParser(description='What state to download')
parser.add_argument('--state', help='What state to download', required=True)
args = vars(parser.parse_args())

base_url = 'https://dataverse.harvard.edu/'
api_token= '9491d91a-c5ac-4612-9ac3-2b866ae7f5c2' # change api token here
api = NativeApi(base_url, api_token)
data_api = DataAccessApi(base_url, api_token)
DOI = "doi:10.7910/DVN/CUWWYJ"
dataset = api.get_dataset(DOI)


files_list = dataset.json()['data']['latestVersion']['files']
for file in files_list:
    filename = file["dataFile"]["filename"]
    file_id = file["dataFile"]["id"]
    persistentId = file["dataFile"]["persistentId"]
    if args['state'] in filename:
        print(file)
        print("File name {}, id {}".format(filename, file_id))
        print(type(file_id))
        response = data_api.get_datafile(str(persistentId))
        print(response.status_code)
        with open(os.path.join(raw_dir, filename), "wb") as f:
            print("Downloading", filename)
            f.write(response.content)
            
        print(os.path.join(raw_dir, filename))
        with zipfile.ZipFile(os.path.join(raw_dir, filename),"r") as zip_ref:
            fname_for_dir = filename[3:].replace('.zip','') # strips the number off of the filename and the .zip
            extract_dir = f"{road_networks}/{args['state']}/Harvard Dataverse/{fname_for_dir}"
            os.makedirs(extract_dir,exist_ok=True)
            zip_ref.extractall(extract_dir)
        