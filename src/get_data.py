"""Setup data from URL"""

from dotenv import load_dotenv
import yaml
import os
import requests
import tarfile


# environment config
load_dotenv()
DATA_URL = os.getenv("DATA_URL")
# local config
with open('config.yaml', 'r') as f:
    ma_config = yaml.safe_load(f)
output_dir = cfg["data"]["local_path"]

# def get_tar_and_extract(url, output_directory):
#     # Téléchargement de l'archive
#     response = requests.get(url)
    
#     # Création du dossier de sortie
#     os.makedirs(output_directory, exist_ok=True)
    
#     # Écriture de l'archive téléchargée dans un fichier temporaire
#     temp_file = os.path.join(output_directory, "temp.tar")
#     with open(temp_file, "wb") as file:
#         file.write(response.content)
    
#     # Extraction de l'archive
#     with tarfile.open(temp_file, "r") as tar:
#         tar.extractall(path=output_directory)
    
#     # Suppression du fichier temporaire
#     os.remove(temp_file)

# # Utilisation de la fonction
# get_tar_and_extract(DATA_URL, output_dir)


if __name__ == "__main__":
    help()
