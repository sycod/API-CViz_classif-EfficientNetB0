"""Setup data from URL"""

import os
import requests
import tarfile
import pandas as pd
import re


def get_tar_and_extract(url, output_directory) -> None:
    """Download, unzip and save datafrom url"""
    # download data
    response = requests.get(url)

    os.makedirs(output_directory, exist_ok=True)
    
    # temporary archive
    temp_file = os.path.join(output_directory, "temp.tar")
    with open(temp_file, "wb") as file:
        file.write(response.content)
    
    # extract archive
    with tarfile.open(temp_file, "r") as tar:
        tar.extractall(path=output_directory)
    
    # remove archive
    os.remove(temp_file)


def create_img_db(img_dir, annot_dir, output_uri) -> pd.DataFrame:
    """Create image and annotation database and export it to CSV file"""
    annot_infos_list = []
    
    # read folder and create dataframe
    breeds_dir = os.listdir(annot_dir)
    # features as they appear in annotation files
    annot_tags = [
        "filename",
        "name",
        "width",
        "height",
        "depth",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "truncated",
        "segment",
        "pose",
        "difficult",
    ]

    # read inside each breed folder
    for breed in breeds_dir:
        annot_list = os.listdir(os.path.join(annot_dir, breed))

        for annot in annot_list:
            # create img URI from information
            img_uri = os.path.join(img_dir, breed, annot + ".jpg")
            # check if corresponding image exists
            if os.path.exists(img_uri):
                pass
            else:
                img_uri = ""

            annot_infos = []
            # read file
            with open(os.path.join(annot_dir, breed, annot), 'r', encoding='utf-8') as f:
                annot_content = f.read()

            # loop over features
            for tag in annot_tags:
                # regex pattern
                pattern = f'<{tag}>(.*?)</{tag}>'
                # store result in a list
                annot_infos.append(re.search(pattern, annot_content, re.DOTALL).group(1))

            # add image URI
            annot_infos.append(img_uri)
            
            # add list to annotations info list
            annot_infos_list.append(annot_infos)

    # store in DF
    db_cols = [
        "ID",
        "class_label",
        "width",
        "height",
        "depth",
        "bb_xmin",
        "bb_ymin",
        "bb_xmax",
        "bb_ymax",
        "truncated",
        "segment",
        "pose",
        "difficult",
        "img_uri",
    ]
    df = pd.DataFrame(annot_infos_list, columns=db_cols)

    # save CSV
    with open(output_uri, "wb") as f:
        df.to_csv(f)

    return df


if __name__ == "__main__":
    help()
