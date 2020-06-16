# RAPIDS Hyperparameter Optimization on Databricks

We make use of MLflow tracking and Hyperopt for performing HPO with RAPIDS on Databricks.

## To-do: add more info about cluster set-up

## Sample Notebook

The sample notebook illustrates how to use MLflow and Hyperopt with RAPIDS to perform HPO experiment and tracking with RandomForestClassifier on Airlines dataset to predict if an airline will be delayed or not. This notebook can be found under `databricks/notebooks/`

Run the `databricks/setup/acquire_data.ipynb` once to upload the data to the DBFS file system, and then run the sample notebook.
