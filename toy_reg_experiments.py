import card
import sys
from card.architectures import RegressionMeanEstimator, RegressionNoiseEstimator
from card.toysamplers import sample_toy_data_by_index
from card import metrics
import torch
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt
from os.path import join, exists
import pandas as pd
from datetime import datetime
import time
from os import makedirs

if __name__ == '__main__':

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
    MODEL_DIRECTORY = join('models', 'toys')
    EXPERIMENTS = ['Linear', 'Quadratic', 'Log-Linear', 'Log-Cubic', 'Sinusoidal', 'Inverse-Sinusoidal', 'Gaussians', 'Circle']

    # For graphical purpose, can be ignored
    # fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4)
    # fig.set_size_inches(16, 9)
    # axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    # line_labels = ['Original Distribution', 'Diffusion', 'Mean Estimator']

    # x_gt = []
    # y_gt = []
    # y_est = []
    # mean = []

    M = 10240
    train_size = M * 4 // 5
    test_size = M - train_size

    epochs = 5000#[100, 100, 100, 1000, 100, 5000, 1000, 100]

    for i in range(1, 9):
        print(f'\nToy Data {i}\n')
        start_exp = time.time()
        k = 0
        scores: pd.DataFrame = None
        scores_path = f'scores/toys/{EXPERIMENTS[i-1]}_scores.csv'
        if exists(scores_path):
            print(f'Loading already measured results for {EXPERIMENTS[i-1]} dataset experiment')
            scores = pd.read_csv(scores_path)
            unmeasured_scores = scores.query('PICP == 0.0 & QICE == 0.0 & RMSE == 0.0')
            if unmeasured_scores.shape[0] == 0:
                # Print results.
                print(f'{EXPERIMENTS[i-1]} scores already measured. Here are mean and standard deviations of the metric:')
                print(f'PICP: mean= {scores.PICP.mean()}, std= {scores.PICP.std()}')
                print(f'QICE: mean= {scores.QICE.mean()}, std= {scores.QICE.std()}')
                print(f'RMSE: mean= {scores.RMSE.mean()}, std= {scores.RMSE.std()}')
                break
            k = unmeasured_scores.index[0]
        else:
            print(f'No measured results for {EXPERIMENTS[i-1]} dataset found. Starting experiment freshly with fold #1.')
            # folds==20 for all datasets, except 5 for Protein and 1 for Year.
            scores = pd.DataFrame(data={
                'PICP': [0.0]*10,
                'QICE': [0.0]*10,
                'RMSE': [0.0]*10,
            })
        print(f'Running {EXPERIMENTS[i-1]} experiment with {10} folds...')


        #########################################
        #           mean estimator              #
        #########################################
        print('Training mean estimator')

        mean_estimator: RegressionMeanEstimator = None
        mean_estimator_path = join(MODEL_DIRECTORY, f'{EXPERIMENTS[i-1]}_mean_estimator.pth')
        try:
            mean_estimator = torch.load(mean_estimator_path)
            print('Pre-trained regression mean estimator loaded.')
        except FileNotFoundError as e:
            print(f'File {mean_estimator_path} not found. Creating and fitting RegressionMeanEstimator...')
            X_me, y_me = sample_toy_data_by_index(M, index=i)
            # Used for standardizing (test) in dataset 4.
            if i == 4:
                means = y_me.mean(dim=0, keepdim=True)
                stds = y_me.std(dim=0, keepdim=True)
                y_me = (y_me - means) / stds
            
            train_dataset_me = TensorDataset(X_me, y_me)
            mean_estimator = RegressionMeanEstimator(1, 1, torch.optim.Adam,
                1e-3, 'test')
            mean_estimator.fit(train_dataset_me)
            torch.save(mean_estimator, mean_estimator_path)


        #########################################
        #           noise estimator             #
        #########################################

        print('\nTraining noise estimator')
        for fold in range(k, 10):
            start_fold = time.time()
            X, y = sample_toy_data_by_index(M, index=i)

            # Used for standardizing (test) in dataset 4.
            if i == 4:
                means = y.mean(dim=0, keepdim=True)
                stds = y.std(dim=0, keepdim=True)
                y = (y - means) / stds
            else: 
                means = 0
                stds = 1

            dataset = TensorDataset(X, y)
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            # Create and train noise estimator model.
            timestmp = datetime.now().strftime("%Y-%m-%d %H:%M")
            noise_estimator: RegressionNoiseEstimator = None
            noise_estimator_path = join(MODEL_DIRECTORY, f'{EXPERIMENTS[i-1]}_noise_estimator_fold-{fold+1}-{timestmp}.pth')

            # Train Noise Estimator.
            print(f'Creating and fitting RegressionNoiseEstimator...')
            noise_estimator = RegressionNoiseEstimator(1, 1, mean_estimator, experiment_name='test')
            optimizer = torch.optim.Adam(noise_estimator.parameters())
            start_RNE_training = time.time()
            noise_estimator.fit(train_dataset, epochs=epochs)#epochs[i-1])

            # Measure RegressionNoiseEstimator training duratation. 
            end_RNE_training = time.time()
            print(f'Regression noise estimator training took {end_RNE_training-start_RNE_training:.2f} seconds.')

            # Save trained model for later inspections.
            torch.save(noise_estimator, noise_estimator_path)




            #########################################
            #             inference                 #
            #########################################
            # print('\nEvaluating model\n')
            # test_infer, y_truth = sample_toy_data_by_index(2000, None, i)
            # labels = noise_estimator.infer(test_infer)

            # if i != 4:
            #     means_est = mean_estimator(test_infer).detach().numpy()
            # # Used for standardizing (test) in dataset 4.
            # else:
            #     means = y_truth.mean(dim=0, keepdim=True)
            #     stds = y_truth.std(dim=0, keepdim=True)
            #     labels = labels * stds + means
            #     means_est = (mean_estimator(test_infer)*stds + means).detach().numpy()

        #     axes[i-1].scatter(test_infer, y_truth, s = .5)
        #     axes[i-1].scatter(test_infer,labels, s = .5)
        #     axes[i-1].scatter(test_infer, means_est,s = .05, color = 'Aqua')
        #     axes[i-1].grid()

        #     x_gt.append(test_infer)
        #     y_gt.append(y_truth)
        #     y_est.append(labels)
        #     mean.append(means_est)

        # fig.suptitle('Scatterplots for toy examples')
        # fig.legend(loc="upper right", labels=line_labels)
        # plt.show()
        
        # Plotting all datasets in separate plots.
        # for i in range(8):
        #     plt.scatter(x_gt[i], y_gt[i], s = 2)
        #     plt.scatter(x_gt[i], y_est[i], s = 2)
        #     plt.scatter(x_gt[i], mean[i], s = 2, color = 'Aqua')
        #     plt.legend(labels = line_labels)
        #     plt.grid()
        #     plt.show()


        #########################################
        #              metrics                  #
        #########################################
            print('Calculating Metrics')    
            PICP, QICE, RMSE = metrics.toy_metrics(noise_estimator, test_dataset, DEVICE, i, means, stds)       
            scores.iloc[fold, :] = PICP, QICE, RMSE

            # Measure fold duration.
            end_fold = time.time()
            print(f'Fold #{fold+1} took {end_fold-start_fold:.2f} seconds.')

            if not exists('scores/toys/'):
                makedirs('scores/toys/')
            scores.to_csv(scores_path, index=False)

        # Print results.
        print(f'PICP: mean= {scores.PICP.mean()}, std= {scores.PICP.std()}')
        print(f'QICE: mean= {scores.QICE.mean()}, std= {scores.QICE.std()}')
        print(f'RMSE: mean= {scores.RMSE.mean()}, std= {scores.RMSE.std()}')

        # Measure experiment duration.
        end_exp = time.time()
        print(f'{EXPERIMENTS[i-1]} experimenten took {end_exp-start_exp:.2f} seconds.')
