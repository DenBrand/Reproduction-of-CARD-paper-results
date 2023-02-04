from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import numpy as np
import pandas as pd
import card.architectures
from card.architectures import RegressionMeanEstimator, RegressionNoiseEstimator
from card.metrics import combined_metrics
from json import load
import argparse
import requests
import shutil
import sys
from tqdm.auto import tqdm
from os import makedirs, getcwd
from os.path import join, exists
import sklearn.datasets
from scipy.io import arff
import zipfile
from datetime import datetime
import time
import json
import warnings # We understand that boston housing dataset will
# be removed in future sk-learn versions and repress its warning.
warnings.filterwarnings('ignore', category=FutureWarning)

parser = argparse.ArgumentParser(
    prog='UCI-Regression-Experiment-Runner'
)
parser.add_argument('-e', '--exp-names', nargs='+', default=[],
    help='List of experiment names to perform. (e.g. -e Concrete Boston Year)')
args = parser.parse_args()

# IF ON LINUX: SET PROCESS NICENESS TO 19.
isWindows = True
try:
    sys.getwindowsversion()
except AttributeError:
    isWindows = False
if not isWindows:
    from os import nice
    nice(19)

DEVICE = card.architectures.DEVICE
TRAIN_TEST_RATIO = 0.9 # corresponds to 90 % training, 10 % test
MODEL_DIRECTORY = join('models', 'regression')
if not exists(MODEL_DIRECTORY): makedirs(MODEL_DIRECTORY)
DATASETS_DIRECTORY = join('datasets', 'regression')

SKLEARN_DATA_LOADERS = {
    'Boston': sklearn.datasets.load_boston,
    'Wine': sklearn.datasets.load_wine
}

def load_dataset(dataset_name: str, src: str, batch_size: int, M: int, input_size: int):
    print(f'Loading {dataset_name} dataset...')
    if src == 'sklearn':
        # Load dataset from sklearn.datasets submodule.
        load_function = SKLEARN_DATA_LOADERS[dataset_name]
        X, y = dataset = load_function(return_X_y=True)
        X = torch.from_numpy(np.float32(X))
        y = torch.from_numpy(np.float32(y))
    else:
        # DATASET NOT LOADED VIA SCIKIT-LEARN
        # Presume file is provided via http.
        assert src.startswith('http'), f'Dataset {dataset_name} download does not '\
            + 'use http or https protocol.'

        # Get filename ending.
        filename = src.split('/')[-1]
        file_type = filename[filename.find('.'):]

        # Ensure dataset directory exists.
        if not exists(join(DATASETS_DIRECTORY, dataset_name)):
            makedirs(join(DATASETS_DIRECTORY, dataset_name))
        if not exists(join(DATASETS_DIRECTORY, dataset_name, dataset_name+file_type)):
            # Download dataset.
            print(f'Dataset {dataset_name} not locally stored yet. Downloading it from:\n{src}')
            with requests.get(src, stream=True) as req:
                # Get content length in bytes.
                total_length = req.headers.get('Content-Length')
                total_length = 0 if not total_length else int(total_length)

                # Progress bar with tqdm.
                with tqdm.wrapattr(req.raw, 'read', total=total_length, desc='') as raw:
                    # Save retrieved file.
                    with open(join(DATASETS_DIRECTORY, dataset_name, dataset_name+file_type), 'wb') as output:
                        shutil.copyfileobj(raw, output)
            print(f'Dataset {dataset_name} downloaded successfully.')

        dataset_path = join(DATASETS_DIRECTORY, dataset_name, dataset_name+file_type)
        if dataset_name not in ('Concrete', 'Energy', 'Kin8nm', 'Naval', 'Power', 'Protein', 'Wine', 'Yacht', 'Year'):
            raise ValueError(f'Unknown dataset {dataset_name}.')
            
        # Read data respective data file.
        data: pd.DataFrame = None
        if dataset_name in ('Concrete', 'Energy'):
            data = pd.read_excel(dataset_path)
        elif dataset_name == 'Kin8nm':
            data = arff.loadarff(dataset_path)
            data = pd.DataFrame(data[0])
        elif dataset_name == 'Naval':
            # Unzip Naval dataset if not done yet.
            if not exists(join(DATASETS_DIRECTORY, dataset_name, 'UCI_CBM_Dataset', 'data.txt')):
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    zip_ref.extractall(join(DATASETS_DIRECTORY, dataset_name))
            print(getcwd())
            print(join(DATASETS_DIRECTORY, dataset_name, 'UCI_CBM_Dataset', 'data.txt'))
            data = pd.read_csv(
                join(DATASETS_DIRECTORY, dataset_name, 'UCI_CBM_Dataset', 'data.txt'),
                delimiter=';'
            )
            print(data)
            print(data.shape)
        elif dataset_name == 'Power':
            # Unzip Power dataset if not done yet.
            if not exists(join(DATASETS_DIRECTORY, dataset_name, 'CCPP', 'Folds5x2_pp.xlsx')):
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    zip_ref.extractall(join(DATASETS_DIRECTORY, dataset_name))
            data = pd.read_excel(join(DATASETS_DIRECTORY, dataset_name, 'CCPP', 'Folds5x2_pp.xlsx'))
        elif dataset_name == 'Protein':
            data = pd.read_csv(dataset_path, delimiter=',')
        elif dataset_name == 'Wine':
            data = pd.read_csv(dataset_path, delimiter=';')
        elif dataset_name == 'Yacht':
            lines = []
            with open(dataset_path, 'r') as file:
                lines = file.readlines()
                lines = [line.replace(' \n', '\n').replace('  ', ' ') for line in lines]
            with open(dataset_path, 'w') as file:
                file.writelines(lines)

            data = pd.read_csv(dataset_path, delimiter=' ', header=None)
        elif dataset_name == 'Year':
            # Unzip Year dataset if not done yet.
            if not exists(join(DATASETS_DIRECTORY, dataset_name, 'YearPredictionMSD.txt')):
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    zip_ref.extractall(join(DATASETS_DIRECTORY, dataset_name))
            data = pd.read_csv(join(DATASETS_DIRECTORY, dataset_name, 'YearPredictionMSD.txt'), header=None)
        else:
            raise ValueError(f'Dataset {dataset_name} could not be handled yet.')

        # Split features and targets.
        if dataset_name in ('Concrete', 'Kin8nm', 'Power', 'Wine', 'Yacht'):
            X, y = data.iloc[:, :-1], data.iloc[:, -1]
        elif dataset_name in ('Energy', 'Naval'):
            X, y = data.iloc[:, :-2], data.iloc[:, -2:]
        elif dataset_name in ('Protein', 'Year'):
            X, y = data.iloc[:, 1:], data.iloc[:, :1]
        else:
            raise ValueError(f'There is no method to split {dataset_name} into X and y yet.')
        X = torch.Tensor(X.values)
        y = torch.Tensor(y.values)

    mean_X = X.mean(dim=0)
    mean_y = y.mean(dim=0)
    std_X = X.std(dim=0)
    std_y = y.std(dim=0)
    mean_X = X.mean(dim=0)
    mean_y = y.mean(dim=0)
    std_X = X.std(dim=0)
    std_y = y.std(dim=0)
    X = (X - mean_X) / std_X
    y = (y - mean_y) / std_y
    
    if y.shape[-1] != 1:
        y = y.unsqueeze(-1) # TODO: Check if this fix really works?

    # Check if input size and number of samples are correct.
    assert input_size == X.shape[1], f'Dataset {dataset_name} has input_size {X.shape[1:]}, but {input_size} were explected.'
    assert M == X.shape[0], f'Dataset {dataset_name} has {X.shape[0]} samples, but M were expected.'
    
    # Split data und build dataloaders.
    M = X.shape[0]
    train_size = int(M * (100*TRAIN_TEST_RATIO) // 100)
    test_size = M - train_size
    dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # TODO: ENSURE SHAPE OF INPUT AND OUTPUT TENSORS FIT TORCH REQUIREMENTS
    print(f'Dataloaders for {dataset_name} dataset built successfully.')

    return train_dataset, test_dataset, mean_X, mean_y, std_X, std_y
            
# SET DATASETS TO PERFORM EXPERIMENT ON HERE.
DATASETS = list(args.exp_names)

if __name__ == '__main__':
    if not DATASETS:
        print('Please provide a list of experiments to run. (For example: "Concrete Boston Year Kin8nm", CTRL+C to cancel this run.)')
        answ = input('Datasets: ')
        DATASETS = answ.strip().split(' ')

    print(f'Running experiments: {str(DATASETS)[1:-1]}...')

    # Read in dataset meta-data.
    with open('UCI_dataset_sources.json', 'r') as file:
        dataset_metadata = load(file)

    for dataset_name in DATASETS:
        start_exp = time.time()
        src, batch_size, M, input_size, folds, output_size = dataset_metadata[dataset_name].values()
        
        # Load already measured results if existent.
        i = 0
        scores: pd.DataFrame = None
        scores_path = f'scores/regression/{dataset_name}_scores.csv'
        if exists(scores_path):
            print(f'Loading already measured results for {dataset_name} dataset experiment')
            scores = pd.read_csv(scores_path)
            unmeasured_scores = scores.query('PICP == 0.0 & QICE == 0.0 & RMSE == 0.0 & NLL == 0.0')
            if unmeasured_scores.shape[0] == 0:
                # Print results.
                print(f'{dataset_name} scores already measured. Here are mean and standard deviations of the metric:')
                print(f'PICP: mean= {scores.PICP.mean()}, std= {scores.PICP.std()}')
                print(f'QICE: mean= {scores.QICE.mean()}, std= {scores.QICE.std()}')
                print(f'RMSE: mean= {scores.RMSE.mean()}, std= {scores.RMSE.std()}')
                print(f'NLL: mean= {scores.NLL.mean()}, std= {scores.NLL.std()}')
                break
            i = unmeasured_scores.index[0]
        else:
            print(f'No measured results for {dataset_name} dataset found. Starting experiment freshly with fold #1.')
            # folds==20 for all datasets, except 5 for Protein and 1 for Year.
            scores = pd.DataFrame(data={
                'PICP': [0.0]*folds,
                'QICE': [0.0]*folds,
                'RMSE': [0.0]*folds,
                'NLL': [0.0]*folds
            })
        print(f'Running {dataset_name} experiment with {folds} folds...')

        # Load mean estimator model if existent, else create and train.
        mean_estimator: RegressionMeanEstimator = None
        mean_estimator_path = join(MODEL_DIRECTORY, f'UCI_{dataset_name}_mean_estimator.pth')
        try:
            mean_estimator = torch.load(mean_estimator_path)
            print('Pre-trained regression mean estimator loaded.')
        except FileNotFoundError as e:
            print(f'File {mean_estimator_path} not found. Creating and fitting RegressionMeanEstimator...')
            train_dataset = load_dataset(dataset_name, src, batch_size, M, input_size)[0]
            mean_estimator = RegressionMeanEstimator(
                input_size,
                output_size,
                torch.optim.Adam,
                1e-3,
                f'RegMeanEst-{dataset_name}-{datetime.now().strftime("%Y-%m-%d %H:%M")}')
            mean_estimator.fit(train_dataset, batch_size=batch_size) # NOTE: The same as for noise estimator training.
            torch.save(mean_estimator, mean_estimator_path)

        for fold in range(i, folds):
            start_fold = time.time()
            datasets = load_dataset(dataset_name, src, batch_size, M, input_size)
            train_dataset, test_dataset, _, mean_y, _, std_y = datasets

            # Create and train noise estimator model.
            timestmp = datetime.now().strftime("%Y-%m-%d %H:%M")
            noise_estimator: RegressionNoiseEstimator = None
            noise_estimator_path = join(MODEL_DIRECTORY, f'UCI_{dataset_name}_noise_estimator_fold-{fold+1}-{timestmp}.pth')

            # Train Noise Estimator.
            print(f'Creating and fitting RegressionNoiseEstimator...')
            noise_estimator = RegressionNoiseEstimator(
                input_size,
                output_size,
                mean_estimator,
                experiment_name=f'RegNoiseEst-{dataset_name}-fold{fold+1}-{timestmp}') # output_size==1 not generalized yet
            optimizer = torch.optim.Adam(noise_estimator.parameters())
            start_RNE_training = time.time()
            noise_estimator.fit(train_dataset, batch_size, epochs=5000)

            # Measure RegressionNoiseEstimator training duratation. 
            end_RNE_training = time.time()
            print(f'Regression noise estimator training took {end_RNE_training-start_RNE_training:.2f} seconds.')

            # Save trained model for later inspections.
            torch.save(noise_estimator, noise_estimator_path)

            # Evaluate noise estimator model with PICP and QICE.
            print('Calculating Metrics')
            PICP, QICE, RMSE, NLL = combined_metrics(
                noise_estimator, test_dataset,
                DEVICE, mean_y, std_y,
                dataset_name=dataset_name)
            scores.iloc[fold, :] = PICP, QICE, RMSE, NLL

            # Measure fold duration.
            end_fold = time.time()
            print(f'Fold #{fold+1} took {end_fold-start_fold:.2f} seconds.')

            if not exists('scores/regression/'):
                makedirs('scores/regression/')
            scores.to_csv(scores_path, index=False)

        # Print results.
        print(f'PICP: mean= {scores.PICP.mean()}, std= {scores.PICP.std()}')
        print(f'QICE: mean= {scores.QICE.mean()}, std= {scores.QICE.std()}')
        print(f'RMSE: mean= {scores.RMSE.mean()}, std= {scores.RMSE.std()}')
        print(f'NLL: mean= {scores.NLL.mean()}, std= {scores.NLL.std()}')

        # Measure experiment duration.
        end_exp = time.time()
        print(f'{dataset_name} experimenten took {end_exp-start_exp:.2f} seconds.')