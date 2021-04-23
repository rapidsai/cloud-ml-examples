# [Process for conversion of mortgage CSV data to Parquet](#anchor-start)
### Obtain the data
For this guide we'll simply assume that we want the entire mortgage dataset. Feel free to adjust according to your needs
and use case.

1. Download the data from the [rapidsai.github.io](https://rapidsai.github.io/demos/datasets/mortgage-data)
1. Follow the `Using data with RAPIDS container` instructions, verifying the correct data structure
    ```
    /rapids/data/mortgage/acq/         <- all acquisition data
    /rapids/data/mortgage/perf/        <- all performance data
    /rapids/data/mortgage/names.csv    <- lender name normalization
    ```

### Upload to your GCS environment.
1. `gsutil -m cp -r names.csv acq perf gs:/[YOUR_MORTGAGE_DATA_PATH]/mortgage_large_csv`

### Convert to Parquet
Once you have landed the CSV dataset into your personal GCP data bucket. Follow the steps defined in the 
[Mortgage Data Conversion Notebook](https://github.com/rapidsai/cloud-ml-examples/blob/main/dask/kubernetes/Mortgage_Data_Conversion.ipynb)