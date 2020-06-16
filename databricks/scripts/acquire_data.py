# Databricks notebook source
import cudf
import numpy as np
import os
from urllib.request import urlretrieve
import gzip

# Set the DBFS path 
data_dir = "/local_path/" # DBFS path to the encompassing folder
file_name = 'airline_20000000.orc' # Stores the name of the downloaded file
orc_name = os.path.join(data_dir, file_name)

def prepare_dataset():

    input_cols = ["Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime", "CRSArrTime",
                  "UniqueCarrier", "FlightNum", "ActualElapsedTime", "Origin", "Dest",
                  "Distance", "Diverted"]

    # Download URL
    url = 'https://rapids-csp.s3-us-west-2.amazonaws.com/data/airline_20000000.orc'
    
    if os.path.isfile(orc_name):
        print(f" > File already exists. Ready to load at {orc_name}")
    else:
        # Ensure folder exists
        os.makedirs(data_dir, exist_ok=True)
        def data_progress_hook(block_number, read_size, total_filesize):
            if (block_number % 1000) == 0:
                print(
                    f" > percent complete: { 100 * ( block_number * read_size ) / total_filesize:.2f}\r",
                    end="",
                )
            return
        urlretrieve(
            url= url,
            filename=orc_name,
            reporthook=data_progress_hook,
        )
        
        print(f" > Download complete {url}")
        
    dataset = cudf.read_orc(orc_name)

    # encode categoricals as numeric
    for col in dataset.select_dtypes(["object"]).columns:
        dataset[col] = dataset[col].astype("category").cat.codes.astype(np.int32)

    # cast all columns to int32
    for col in dataset.columns:
        dataset[col] = dataset[col].astype(np.float32)  # needed for random forest

    # put target/label column first [ classic XGBoost standard ]
    output_cols = ["ArrDelayBinary"] + input_cols

    dataset = dataset.reindex(columns=output_cols)
    return dataset


df = prepare_dataset()

df.to_parquet("/local_path/airline_20000000.parquet")