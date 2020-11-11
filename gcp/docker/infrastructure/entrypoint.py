import argparse
import random
import gcsfs
import logging
import hypertune
import json
import sys

from typing import (List,
                    Set,
                    Dict,
                    Tuple,
                    Optional)

# RayTune + Ax elements
from ray import tune
from ray.tune import track
from ray.tune.suggest.ax import AxSearch
from ax import ParameterType, optimize
from ax.service.ax_client import AxClient

logger = logging.getLogger(tune.__name__)
logger.setLevel(level=logging.CRITICAL)

from rapids_lib import RapidsCloudML

default_sagemaker_paths = {
    'base': '/opt/ml',
    'code': '/opt/ml/code',
    'data': '/opt/ml/input',
    'train_data': '/opt/ml/input/data/training',
    'hyperparams': '/opt/ml/input/config/hyperparameters.json',
    'model': '/opt/ml/model',
    'output': '/opt/ml/output',
}


def ax_train_proxy(model_params, config_params, ax_params):
    rcml = RapidsCloudML(cloud_type=config_params['cloud_type'],
                         model_type=config_params['model_type'],
                         compute_type=f"single-{args.compute_type}",
                         CSP_paths=config_params['paths'])

    # environment check
    rcml.environment_check()

    # ingest data [ post pre-processing ]
    dataset, col_labels, y_label, ingest_time = rcml.load_data(filename=config_params['dataset_filename'])
    rcml.query_memory()

    # classification objective requires int32 label for cuml random forest
    dataset[y_label] = dataset[y_label].astype('int32')

    accuracy_per_fold = []
    train_time_per_fold = []
    infer_time_per_fold = []
    split_time_per_fold = []
    global_best_model = None
    global_best_test_accuracy = 0

    model_params["max_depth"] = ax_params["max_depth"]
    model_params["max_features"] = ax_params["max_features"]
    model_params["n_estimators"] = ax_params["n_estimators"]

    # optional cross-validation w/ model_params['n_train_folds'] > 1
    for i_train_fold in range(config_params['CV_folds']):
        print(f"STARTING TRAINING FOLD {i_train_fold}", flush=True)
        rcml.log_to_file(f"\n CV fold {i_train_fold} of {config_params['CV_folds']}\n")

        # split data
        X_train, X_test, y_train, y_test, split_time = rcml.split_data(dataset=dataset,
                                                                       y_label=y_label,
                                                                       random_state=i_train_fold,
                                                                       shuffle=True)
        split_time_per_fold += [round(split_time, 4)]

        # train model
        trained_model, training_time = rcml.train_model(X_train, y_train, model_params)
        train_time_per_fold += [round(training_time, 4)]

        # evaluate perf
        test_accuracy, infer_time = rcml.evaluate_test_perf(trained_model, X_test, y_test)
        accuracy_per_fold += [round(test_accuracy, 4)]
        infer_time_per_fold += [round(infer_time, 4)]

        # update best model [ assumes maximization of perf metric ]
        if test_accuracy > global_best_test_accuracy:
            global_best_test_accuracy = test_accuracy
            global_best_model = trained_model

        rcml.log_to_file(f'\n accuracy per fold    : {accuracy_per_fold} \n')
        rcml.log_to_file(f'\n train-time per fold  : {train_time_per_fold} \n')
        rcml.log_to_file(f'\n infer-time per fold  : {infer_time_per_fold} \n')
        rcml.log_to_file(f'\n split-time per fold  : {split_time_per_fold} \n')

    track.log(accuracy=global_best_test_accuracy)

# Service based example
def raytune_ax_train(model_params: dict, config_params: dict):
    depth = [int(d) for d in config_params['ht_depth_range'].split(',')]
    features = [float(d) for d in config_params['ht_features_range'].split(',')]
    estimators = [int(d) for d in config_params['ht_est_range'].split(',')]
    experiments = config_params['ht_experiments']
    ax = AxClient(enforce_sequential_optimization=False)

    ax.create_experiment(
        name="hpo_experiment",
        parameters=[
            {"name": "max_depth", "type": "range", "bounds": depth, "parameter_type": ParameterType.INT},
            {"name": "max_features", "type": "range", "bounds": features, "parameter_type": ParameterType.FLOAT},
            {"name": "n_estimators", "type": "range", "bounds": estimators, "parameter_type": ParameterType.INT}
        ],
        objective_name="accuracy",
        minimize=False
    )

    tune.run(
        run_or_experiment=lambda parameters: ax_train_proxy(model_params=model_params,
                                                            config_params=config_params,
                                                            ax_params=parameters),
        num_samples=experiments,
        search_alg=AxSearch(ax),  # Note that the argument here is the `AxClient`.
        verbose=1,  # Set this level to 1 to see status updates and to 2 to also see trial results.
        # To use GPU, specify: resources_per_trial={"gpu": 1}.
        resources_per_trial={"gpu": 1} if ('GPU' in config_params['compute']) else {"cpu": 8}
    )

    print(f"FINISHED RAY TUNE RUNE", flush=True)

    best_parameters, best_values = ax.get_best_parameters()
    means, covariances = best_values
    print("Ax Optimization Results:", flush=True)
    print(best_parameters, flush=True)
    print(best_values, flush=True)

    return means['accuracy']


# Loop based example
# Not currently used
def ax_train(rcml, model_params: dict, config_params: dict):
    # ----------------------------------------------------------------------------------------------------
    # cross-validation folds
    # ----------------------------------------------------------------------------------------------------
    global_best_model = None
    global_best_test_accuracy = 0

    parameters = [
        {
            "name": "max_depth",
            "type": "range",
            "bounds": [9, 17],
            "parameter_type": ParameterType.INT
        },
        {
            "name": "max_features",
            "type": "range",
            "bounds": [0.20, 0.6],
            "parameter_type": ParameterType.FLOAT
        },
        {
            "name": "n_estimators",
            "type": "range",
            "bounds": [100, 200],
            "parameter_type": ParameterType.INT
        }
    ]

    best_parameters, best_values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=lambda params: ax_train_proxy(model_params=model_params,
                                                          config_params=config_params,
                                                          ax_params=params),
        minimize=False,
        total_trials=10,
        objective_name='accuracy',
    )

    print("Ax Optimization Results:")
    print(best_parameters)
    print(best_values)

    # save model
    # rcml.save_best_model(global_best_model)

    # ----------------------------------------------------------------------------------------------------
    # TODO: baseline [ single-node - multi-GPU w/ dask ]
    # ----------------------------------------------------------------------------------------------------
    return global_best_model, global_best_test_accuracy


def _train(model_params, config_params):
    rcml = RapidsCloudML(cloud_type=config_params['cloud_type'],
                         model_type=config_params['model_type'],
                         compute_type=f"single-{args.compute_type}",
                         CSP_paths=config_params['paths'])

    # environment check
    rcml.environment_check()

    # ingest data [ post pre-processing ]
    dataset, col_labels, y_label, ingest_time = rcml.load_data(filename=config_params['dataset_filename'])
    rcml.query_memory()

    # classifier expects input data to be of type float32
    dataset = dataset.astype('float32')
    # classification objective requires int32 label for cuml random forest
    dataset[y_label] = dataset[y_label].astype('int32')

    # ----------------------------------------------------------------------------------------------------
    # cross-validation folds
    # ----------------------------------------------------------------------------------------------------
    global_best_model = None
    global_best_test_accuracy = 0
    accuracy_per_fold = []
    train_time_per_fold = []
    infer_time_per_fold = []
    split_time_per_fold = []

    # optional cross-validation w/ model_params['n_train_folds'] > 1
    for i_train_fold in range(config_params['CV_folds']):
        rcml.log_to_file(f"\n CV fold {i_train_fold} of {config_params['CV_folds']}\n")

        # split data
        X_train, X_test, y_train, y_test, split_time = rcml.split_data(dataset=dataset,
                                                                       y_label=y_label,
                                                                       random_state=i_train_fold,
                                                                       shuffle=True)
        split_time_per_fold += [round(split_time, 4)]

        # train model
        trained_model, training_time = rcml.train_model(X_train, y_train, model_params)
        train_time_per_fold += [round(training_time, 4)]

        # evaluate perf
        test_accuracy, infer_time = rcml.evaluate_test_perf(trained_model, X_test, y_test)
        accuracy_per_fold += [round(test_accuracy, 4)]
        infer_time_per_fold += [round(infer_time, 4)]

        # update best model [ assumes maximization of perf metric ]
        if test_accuracy > global_best_test_accuracy:
            global_best_test_accuracy = test_accuracy
            global_best_model = trained_model

    rcml.log_to_file(f'\n accuracy per fold    : {accuracy_per_fold} \n')
    rcml.log_to_file(f'\n train-time per fold  : {train_time_per_fold} \n')
    rcml.log_to_file(f'\n infer-time per fold  : {infer_time_per_fold} \n')
    rcml.log_to_file(f'\n split-time per fold  : {split_time_per_fold} \n')

    return global_best_test_accuracy


def train(model_params, config_params):
    """
    Parameters
    ----------
    model_params
    config_params
    do_ax_hpo

    Returns
    -------

    """
    #rcml = RapidsCloudML(cloud_type=config_params['cloud_type'],
    #                     model_type=config_params['model_type'],
    #                     compute_type=f"single-{args.compute_type}",
    #                     CSP_paths=config_params['paths'])

    ## environment check
    #rcml.environment_check()

    ## ingest data [ post pre-processing ]
    #dataset, col_labels, y_label, ingest_time = rcml.load_data(filename=config_params['dataset_filename'])
    #rcml.query_memory()

    ## classification objective requires int32 label for cuml random forest
    #dataset[y_label] = dataset[y_label].astype('int32')

    # ----------------------------------------------------------------------------------------------------
    # cross-validation folds
    # ----------------------------------------------------------------------------------------------------
    global_best_model = None
    global_best_test_accuracy = 0

    if (config_params['do_ax_hpo']):
        global_best_test_accuracy = raytune_ax_train(config_params=config_params, model_params=model_params)
    else:
        global_best_test_accuracy = _train(config_params=config_params, model_params=model_params)

    # save model
    # rcml.save_best_model(global_best_model)

    # ----------------------------------------------------------------------------------------------------
    # TODO: baseline [ single-node - multi-GPU w/ dask ]
    # ----------------------------------------------------------------------------------------------------
    return global_best_model, global_best_test_accuracy


def gcp_path_setup(args):
    # TODO: Ad-hoc, needs to be more generic.
    hyperpath = '/opt/gcp_rapids/hyperparameters.json'
    hyperdict = {}
    for key, val in args.__dict__.items():
        if (key.startswith('hpo')):
            new_key = key.replace('hpo-', '')
            hyperdict[new_key] = val

    with open(hyperpath, 'w') as fpw:
        fpw.write(json.dumps(hyperdict, indent=4, sort_keys=True))

    paths = {
        'train_data': args.data_input_path,
        'hyperparams': hyperpath,
        'model': f'{args.data_output_path}/model',
        'output': f'{args.data_output_path}/output',
    }

    return paths


def aws_path_setup():
    return default_sagemaker_paths


def azure_path_setup():
    return {}


def main(args):
    paths = {}
    if (args.cloud_type.lower() == "gcp"):
        paths = gcp_path_setup(args)
    elif (args.cloud_type.lower() in ("aws")):
        paths = aws_path_setup(args)
    elif (args.cloud_type.lower() in ("azure")):
        paths = azure_path_setup(args)

    config_params = {}
    config_params['CV_folds'] = args.cv_folds
    config_params['compute'] = args.compute_type
    config_params['dataset'] = 'airline'
    config_params['dataset_filename'] = args.data_name
    config_params['cloud_type'] = args.cloud_type
    config_params['model_type'] = args.model_type
    config_params['num_samples'] = args.num_samples
    config_params['paths'] = paths
    config_params['do_ax_hpo'] = args.do_ax_hpo
    config_params['ht_est_range'] = args.ht_est_range
    config_params['ht_depth_range'] = args.ht_depth_range
    config_params['ht_features_range'] = args.ht_features_range
    config_params['ht_experiments'] = args.ht_experiments

    if ('RandomForest' in args.model_type):
        model_params = {
            'max_depth': args.hpo_max_depth,
            'max_features': args.hpo_max_features,
            'n_bins': args.hpo_num_bins,
            'n_estimators': args.hpo_num_est,
            'seed': random.random(),
            # 'seed': 0
        }
    elif ('XGBoost' in args.model_type):
        model_params = {
            'alpha': args.hpo_alpha,
            'gamma': args.hpo_gamma,
            'lambda': args.hpo_lambda,
            'learning_rate': args.hpo_lr,
            'max_depth': args.hpo_max_depth,
            'num_boost_round': args.hpo_num_boost_round,
            'random_state': 0,
            'tree_method': 'gpu_hist' if ('GPU' in config_params['compute']) else 'hist'
        }

    model, accuracy = train(model_params=model_params, config_params=config_params)

    if (args.cloud_type.lower() in ("gcp",) and args.do_hpo):
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='hpo_accuracy',
            metric_value=accuracy)


if (__name__ in ("__main__",)):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud-type', default='AWS')
    parser.add_argument('--compute-type', default='GPU')
    parser.add_argument('--data-input-path')
    parser.add_argument('--data-output-path')
    parser.add_argument('--data-name', default='airline_10000000.orc')
    parser.add_argument('--do-hpo', action="store_true", default=False)
    parser.add_argument('--do-ax-hpo', action="store_true", default=False)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--hpo-alpha', default=0.0, type=float)
    parser.add_argument('--hpo-gamma', default=0.0, type=float)
    parser.add_argument('--hpo-lambda', default=1.0, type=float)
    parser.add_argument('--hpo-lr', default=0.3, type=float)
    parser.add_argument('--hpo-max-depth', default=16, type=int)
    parser.add_argument('--hpo-max-features', default=1.0, type=float)
    parser.add_argument('--hpo-num-bins', default=64, type=int)
    parser.add_argument('--hpo-num-boost-round', default=100, type=int)
    parser.add_argument('--hpo-num-est', default=10, type=int)
    parser.add_argument('--ht-depth-range', default="9,17", type=str)
    parser.add_argument('--ht-est-range', default="100,200", type=str)
    parser.add_argument('--ht-features-range', default="0.2,0.6", type=str)
    parser.add_argument('--ht-experiments', default=10, type=int)
    parser.add_argument('--num-samples', default=4, type=int)
    parser.add_argument('--cv-folds', default=1, type=int)
    parser.add_argument('--job-dir')
    parser.add_argument('--model-type', default="XGBoost", choices=['RandomForest', 'XGBoost'])
    parser.add_argument('--train', action="store_true")

    args = parser.parse_args()

    main(args)

    sys.exit(0)
