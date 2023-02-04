# Reproducibility Study of "CARD: Classification and Regression Diffusion Models"

In this repository we publish our reproducibility project on the "CARD: Classification and Regression Models" paper (https://arxiv.org/abs/2206.07275).

## Requirements

To install requirements run:

```setup
pip install -r requirements.txt
```

If you wish to run our code on a NVIDIA GPU with CUDA support, make sure beforehand that the CUDA Toolkit is installed.

## Training & Evaluation of UCI experiments to reproduce the reported PICP, QICE, RMSE, and NLL scores

To train and directly evaluate the model(s):
1. Make sure your current working directory is the top level project directory (containing the UCI_reg_experiments.py script)
2. If you installed the requirements in a seperate environment, make sure it is activated
3. Run the following command with <dataset-name> replaced by the name of the dataset you wish the experiment on ("Boston", "Concrete", "Energy", ...):

```train-and-eval-uci
python UCI_reg_experiments.py -e <dataset-name>
```

Firstly, the application will print out the memory usage of CUDA GPUs found on the machine, giving the user the opportunity to select the GPU that shall be utilized during the experiment. If there is none or the requirements for its use are not met, the CPU is selected automatically. Next up, the program will try to load the pre-trained mean estimator for the selected experiment, if no such model is stored in the directory provided for this purpose (stored at './models/regression/UCI_<dataset-name>_mean_estimator.pth'), a new one will be created and trained. Now that a mean estimator exists, which can be utilized by the noise estimators, the program enters its main loop, each iteration performing one "fold", that is one cycle of data loading/preparation, noise estimator instantiation and training as well as evaluation of PICP, QICE, RMSE and NLL metric scores. At the end of each fold a new row will be written into a CSV file ('./scores/regression/<dataset-name>_scores.csv'), progressively accumulating the measured results. At the end of the last fold, the application will output the mean values as well as the standard deviations of all four metrics.<br/>
<br/>
Since the program, in the presence of a CSV file with already calculated rows, continues directly with the fold at which it was interrupted the last time, the same command can be used to either resume the interrupted experiment or, if the CSV file is already filled with the necessary number of metric measurements, directly re-calculate the mean and the standard deviation and print them into the console.

## Training & Evaluation of toy example experiments to reproduce the reported PICP, QICE, and RMSE scores

To train and directly evaluate the model(s) follow the same steps as for the UCI experiments above, but use this command instead:

```train-and-eval_toy
python toy_reg_experiments.py
```

This script works mostly similar to the UCI_reg_experiments.py script and will therefore also ask which GPU to use, if there is one. It will also write its results into CSV files (this time placed at './scores/toys/<distribution-name>_scores.csv') and loads potentially existing pre-trained models from 'models/toy/<distribution-name>_mean_estimator.pth'. This program does not need to be told which experiment to run, as it simply runs all toy example experiments in sequence. For multi-modal data distributions, the calculated RMSE score should be ignored, as this metric makes no sense for multi-modal distributions. As with the UCI experiments, the means and standard deviations are output to the console after the last fold (here 10 for every distribution) is completed.

## Pre-trained Models

The pre-trained models for UCI experiments that we provide are located in the './models/regression/' directory, while the toy example models can be found in the './models/toy/' folder.
We only provide pre-trained models for mean estimators because to provide all noise estimators we would have had to keep and upload 146 models for the UCI experiments and 80 for the toy experiments.
However, all mean estimators will be found in the folders already mentioned.
If you wish to generate new mean estimators it would be sufficient to rename the corresponding PTH file.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
