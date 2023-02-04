import numpy as np
import torch
import card.toysamplers as ts
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch.nn.functional as F
import wandb
from datetime import datetime
import pandas as pd
import subprocess
import sys
from io import BytesIO

# NOTE: Antithetic sampling?

def select_gpu():
    gpu_stats = subprocess.check_output(['nvidia-smi', '--format=csv', '--query-gpu=memory.used,memory.free'])
    gpu_df = pd.read_csv(BytesIO(gpu_stats), names=['memory_used', 'memory_free'], skiprows=1)
    print(f'GPU usages:\n{gpu_df}')
    try:
        answ = input(f'Which GPU should be used? {list(gpu_df.index.values)}: ')
        answ = int(answ)
    except ValueError:
        print(f'{answ} is no valid choice. Valid choices would be: {list(gpu_df.index.values)}\nExiting program.')
        sys.exit()
    if int(answ) not in tuple(gpu_df.index.values):
        print(f'{answ} is no valid choice. Valid choices would be: {list(gpu_df.index.values)}\nExiting program.')
        sys.exit()
    return int(answ)

# Set constants
SEED = 1337 # NOTE: Just a test value
DEVICE = None
if torch.cuda.is_available():
    DEVICE = torch.device('cuda', select_gpu())
else:
    DEVICE = torch.device('cpu')

class RegressionMeanEstimator(nn.Module):
    def __init__(self, input_size: int, output_size: int, optim_class, lr: float, experiment_name: str = None):
        super().__init__()

        if experiment_name:
            self.experiment_name = experiment_name

        self.linear = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, output_size)
        )

        self.initial_lr = lr
        self.optim = optim_class(self.parameters(), lr=lr)

    def forward(self, x):
        res = self.linear(x)
        if res.shape[-1] != 1:
            res = res.unsqueeze(-1)
        return res

    def train_one_epoch(self, train_dataloader, loss_fn, log=False):
        total_loss = 0.
        for x, y in train_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Compute prediction error.
            preds = self.forward(x)
            loss = loss_fn(preds, y)
            total_loss += loss

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        if log:
                wandb.log({'epoch_loss': total_loss})

    def fit(self,
            dataset: TensorDataset,
            batch_size=256,
            loss_fn=nn.MSELoss(),
            patience=50,
            epochs=1000,
            verbose=True):
        # Weight reset, use before training on complete dataset after optimal num of epochs was found
        def weight_reset(layer):
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.apply(weight_reset)

        # Load model to device.
        self.to(DEVICE)

        # Split data set into train and val set.
        M = len(dataset)
        train_size = M * 3 // 5
        val_size = M-train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_set, batch_size=batch_size)
        val_dataloader = DataLoader(val_set, batch_size=val_size)
        complete_dataloader = DataLoader(dataset, batch_size=batch_size)

        # Training.
        min_val_loss = np.inf
        current_patience = patience
        optim_epoch_num = -1
        converged = False
        for epoch in range(epochs):
            # Train for one epoch.
            self.train()
            self.train_one_epoch(train_dataloader, loss_fn)
            self.eval()

            # Epoch terminated, now validate.
            for x, y in val_dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = self.forward(x)
                val_loss = loss_fn(preds, y)
                    
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                current_patience = patience
                optim_epoch_num = epoch+1
            else:
                current_patience -= 1

            if verbose:
                print(f'val_loss: {val_loss:>4f}, patience: {current_patience:>2d}, batch: [{epoch:>5d}/{epochs:>5d}]')
        
            if current_patience == 0:
                converged = True
                break

        # Initialize Weights and Biases run.
        wandb.init(
            project='CARDboard',
            entity='denbrand',
            name=self.experiment_name,
            mode='offline'
            )

        # Did it converge?
        if not converged:
            print(f'Training did not converge after {epochs} epochs (patience did not reach 0). Training on complete dataset (=train_set+val_set) for {epochs} epochs...')
            self.apply(weight_reset)

            wandb.config = {
                'batch_size': batch_size,
                'initial_lr': self.initial_lr,
                'patience': patience,
                'epochs': epochs
            }

            for epoch in range(epochs):
                self.train()
                self.train_one_epoch(complete_dataloader, loss_fn, log=True)
                self.eval()
        else:
            # Train for empirically optimal number of epochs.
            print(f'Training converged after {optim_epoch_num} epochs.')
            self.apply(weight_reset)

            wandb.config = {
                'batch_size': batch_size,
                'learning_rate': self.initial_lr,
                'patience': patience,
                'epochs': optim_epoch_num
            }

            for epoch in range(optim_epoch_num):
                self.train()
                self.train_one_epoch(complete_dataloader, loss_fn, log=True)
                self.eval()

        wandb.finish()

        print(f'Validating final mean estimator model...')
        for x, y in val_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = self.forward(x)
            val_loss = loss_fn(preds, y)
        print(f'Final loss on validation set: {val_loss:>4f}')

class RegressionNoiseEstimator(nn.Module):
    def __init__(self, input_size, output_size, mean_estimator: RegressionMeanEstimator, T: int=1000, experiment_name: str = None):
        super().__init__()

        if experiment_name:
            self.experiment_name = experiment_name

        # mean estimator
        self.f = mean_estimator
        params = list(self.f.parameters()) 
        for s in params:
            s.requires_grad=False        

        # Time step number T and diffusion schedule alpha_bar with
        # alpha_bar_t = prod_{i=1}^t (1-beta_i)
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, self.T)
        self.alpha_bar = 1 - self.beta
        for t in range(1, T):
            self.alpha_bar[t] = self.alpha_bar[t-1] * self.alpha_bar[t]

        # Actual network
        # layer l_1
        self.g_1a = nn.Linear(input_size+2*output_size, 128)
        self.g_1b = nn.Linear(1, 128)
        self.softplus_1 = nn.Softplus()
        # layer l_2
        self.g_2a = nn.Linear(128, 128)
        self.g_2b = nn.Linear(1, 128)
        self.softplus_2 = nn.Softplus()
        # layer l_3
        self.g_3a = nn.Linear(128, 128)
        self.g_3b = nn.Linear(1, 128)
        self.softplus_3 = nn.Softplus()
        # output layer
        self.g_4 = nn.Linear(128, output_size)
        
    def forward(self, x, y, f_x, t) -> torch.Tensor:
        
        while len(y.shape) > len(x.shape):
            y = y.squeeze(-1)
            f_x = f_x.squeeze(-1)

        # scale t down into interval [0, 1]
        t = t / self.T
        # l_1
        res = torch.cat((x, y, f_x), dim=1)   # concatenation
        res = self.g_1a(res)
        res = res * self.g_1b(t)
        res = self.softplus_1(res)
        # l_2
        res = self.g_2a(res)
        res = res * self.g_2b(t)
        res = self.softplus_1(res)
        # l_3
        res = self.g_3a(res)
        res = res * self.g_3b(t)
        res = self.softplus_1(res)
        # l_4
        res = self.g_4(res)
        # Adjust output dimension
        if res.shape[-1] != 1:
            res = res.unsqueeze(-1)
        return res

    def fit(self,
            dataset: TensorDataset,
            batch_size=256,
            learning_rate=1e-3,
            epochs=1000,
            T=1000):
        # TODO: Include exponentially weighted moving averages
        # on model paramteres with decay of 0.9999 later.

        # Load model to device.
        self.to(DEVICE)

        # Build ADAM optimizer and Loss function
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True)
        loss_fn = nn.MSELoss()

        # Build iterator for test data.
        batches = DataLoader(dataset, batch_size=batch_size)
        
        # Setup finished, enter train mode.
        self.train()

        # Initialize Weights and Biases run.
        wandb.init(project='CARDboard',
            entity='denbrand',
            name=self.experiment_name,
            mode='offline')

        wandb.config = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'T': T
        }

        # Main train loop
        for epoch in range(epochs):
            # Train for one epoch.
            lossTotal = 0

            # Draw batch.
            for x, y_0 in batches:

                # Zero gradients for every batch.
                optim.zero_grad()

                # Load batch into VRAM.
                x, y_0 = x.to(DEVICE), y_0.to(DEVICE)

                # Get mean estimations.
                f_x = self.f(x)

                # Draw t uniformely in {1, ..., T}.
                t = np.random.randint(1, T+1)

                # Draw noises from standard normal distribution.
                eps = torch.normal(0, 1, y_0.shape)
                eps = eps.to(DEVICE)

                # Compute noise estimation loss L_epsilon.
                y_arg = torch.sqrt(self.alpha_bar[t-1]) * y_0
                y_arg += torch.sqrt(1 - self.alpha_bar[t-1]) * eps
                y_arg += (1 - torch.sqrt(self.alpha_bar[t-1])) * f_x
                t = torch.Tensor([[t]])
                x, y_arg, f_x, t = x.to(DEVICE), y_arg.to(DEVICE), f_x.to(DEVICE), t.to(DEVICE) # Load input to device.
                eps_theta = self.forward(x, y_arg, f_x, t)

                # Perform optimization step.
                L_eps = loss_fn(eps_theta, eps)
                lossTotal += L_eps
                L_eps.backward()
                optim.step()

            # Track epoch loss.
            wandb.log({'epoch_loss': lossTotal})

            # print(f'Epoch {epoch+1:>3d}: loss = {lossTotal / len(batches):>4f}, real noise: {eps[0][0]: >9f}, estimated noise: {eps_theta[0][0]:>9f}, last t = {t:>4d}')        
            if((epoch+1) % 128 == 0):
                print(f'Epoch {epoch+1:>3d}: loss = {lossTotal/len(batches):>4f}')
        print(f'Final loss for the noise estimator: {lossTotal/len(batches):>4f}')
        wandb.finish()

    def infer(self, x):
        # Set to evaluation mode.
        self.eval()
        self.to(DEVICE)
        self.f.to(DEVICE)
        with torch.no_grad():
            # Calculate y_T for as starting point.
            x = x.to(DEVICE)
            f_x = self.f(x)
            y_t = torch.normal(f_x, 1)
            
            # Iterating over all timesteps {T-1, ..., 0}.
            for t in range(self.T-1, -1, -1):

                # Calculating y_hat_0 (here just y_0).
                t_arg = torch.Tensor([[t+1]])
                x, y_t, f_x, t_arg = x.to(DEVICE), y_t.to(DEVICE), f_x.to(DEVICE), t_arg.to(DEVICE)
                y_0 = self.forward(x, y_t, f_x, t_arg)
                y_0 *= -torch.sqrt(1 - self.alpha_bar[t])
                y_0 -= (1 - np.sqrt(self.alpha_bar[t])) * f_x
                y_0 += y_t
                y_0 /= torch.sqrt(self.alpha_bar[t])
                
                # Calculating y_{t-1}.
                if t == 0:
                    y_t = y_0
                else:
                    # Draw z.
                    z = torch.normal(0, 1, y_t.shape)
                    z = z.to(DEVICE)

                    # Gamma terms.
                    divisor = 1 - self.alpha_bar[t]
                    gamma_0 = self.beta[t] * torch.sqrt(self.alpha_bar[t-1]) / divisor
                    gamma_1 = (1 - self.alpha_bar[t -1]) * torch.sqrt(1-self.beta[t]) / divisor
                    gamma_2 = (torch.sqrt(self.alpha_bar[t]) - 1) * (torch.sqrt(1-self.beta[t]) + torch.sqrt(self.alpha_bar[t-1])) / divisor
                    gamma_2 += 1

                    # Beta tilde.
                    beta_tilde_sqrt = torch.sqrt((1 - self.alpha_bar[t-1]) * self.beta[t] / divisor)
                    beta_tilde_sqrt = beta_tilde_sqrt.to(DEVICE)

                    # Combining everything to y_(t-1).
                    y_t = gamma_1 * y_t
                    y_t += gamma_0 * y_0
                    y_t += gamma_2 * f_x
                    y_t += beta_tilde_sqrt * z
        return y_t

class ResNetBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel, stride) -> None:
        super().__init__()
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        self.stride = stride
        self.conv1 = nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding = 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(outputChannel)
        self.conv2 = nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding = 1, stride= 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputChannel)

    def forward(self, x):
        if self.stride != 1 or self.inputChannel != self.outputChannel:
            layer = nn.Conv2d(self.inputChannel, self.outputChannel, 1, self.stride, bias = False)
            layer = layer.to(DEVICE)
            identity = layer(x)
            #identity = nn.BatchNorm2d(self.outputChannel, identity)
        else:
            identity = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.bn2(output)
        output = output + identity  
        output = self.bn2(output)
        return F.relu(output)

class ClassificationPretrainedModel(nn.Module): #3 kernel am Anfang, kein max pool
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding = 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxPool = nn.MaxPool2d(2, padding = 0, stride = 2)
        self.conv2_1 = ResNetBlock(64, 64, stride=1)
        self.conv2_2 = ResNetBlock(64, 64, stride=1)
        self.conv3_1 = ResNetBlock(64, 128, stride=2)
        self.conv3_2 = ResNetBlock(128, 128, stride=1)
        self.conv4_1 = ResNetBlock(128, 256, stride=2)
        self.conv4_2 = ResNetBlock(256, 256, stride=1)
        self.conv5_1 = ResNetBlock(256, 512, stride=2)
        self.conv5_2 = ResNetBlock(512, 512, stride=1)
        self.averagePool = nn.AvgPool2d(4, stride = 4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        output = self.conv1(x) 
        output = self.bn1(output)
        output = F.relu(output)
       # output = self.maxPool(output)
        output = self.conv2_1(output)
        output = self.conv2_2(output)
        output = self.conv3_1(output)
        output = self.conv3_2(output)
        output = self.conv4_1(output)
        output = self.conv4_2(output)
        output = self.conv5_1(output)
        output = self.conv5_2(output)
        output = self.averagePool(output)
        output = torch.squeeze(output)
        output =  self.fc(output)
        return F.softmax(output, dim = 1)

    def train_one_epoch(self, train_dataloader, optim, loss_fn=nn.MSELoss(), verbose = False):
        counter = 0

        for batch in train_dataloader:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)

                # Compute prediction error.
                preds = self.forward(x)
                loss = loss_fn(preds.float(), y.float())

                # Backpropagation
                optim.zero_grad()
                loss.backward()
                optim.step()
                counter += 1     
                if verbose and (counter % 10 == 0 or counter  == 1): 
                    print(f'batch = {counter}/{len(train_dataloader)}, loss = {loss}')

    def fit(self,
            dataloader: DataLoader,
            optimizer,
            batch_size=256,
            loss_fn=nn.MSELoss(),
            epochs=10,
            verbose=True):


        # Load model to device.
        self.to(DEVICE)

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}')
            self.train()
            self.train_one_epoch(dataloader, optimizer, loss_fn = loss_fn, verbose = verbose)
            self.eval()
        
                

    def test(self, test_dataloader, verbose = False):
        # Load model to device.
        self.to(DEVICE)
        print('Testing')
        correct = 0
        counter = 0
        for batch in test_dataloader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            counter += 1

            preds = self.forward(x)
            max = torch.max(preds, 1)
            true = torch.max(y, 1)
            correct += torch.count_nonzero(true.indices == max.indices)
            if verbose:
                print(f'correct {correct}/ {min(256*counter, 10000)}')
        print(f'final accuracy = {correct / 10000}')


class ClassificationNoiseEstimator(nn.Module):
    def __init__(self, input_size, num_classes, preTrained: ClassificationPretrainedModel, T: int=1000):
        super().__init__()

        self.num_classes = num_classes
        # Pre Trained Classificator
        self.f = preTrained
        params = list(self.f.parameters()) 
        for s in params:
            s.requires_grad=False        

        # Time step number T and diffusion schedule alpha_bar with
        # alpha_bar_t = prod_{i=1}^t (1-beta_i)
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, self.T)
        self.alpha_bar = 1 - self.beta
        for t in range(1, T):
            self.alpha_bar[t] = self.alpha_bar[t-1] * self.alpha_bar[t]

        # Actual network

        # Input encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.BatchNorm1d(4096),
            nn.Softplus(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Softplus(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096)
        )
        # layer l_1
        self.g_1y = nn.Linear(2* self.num_classes, 4096)
        self.g_1b = nn.Linear(1, 4096)
        self.bn_1 = nn.BatchNorm1d(4096)
        self.softplus_1 = nn.Softplus()
        # layer l_2
        self.g_2y = nn.Linear(4096, 4096)
        self.g_2b = nn.Linear(1, 4096)
        self.bn_2 = nn.BatchNorm1d(4096)
        self.softplus_2 = nn.Softplus()
        # layer l_3
        self.g_3y = nn.Linear(4096, 4096)
        self.g_3b = nn.Linear(1, 4096)
        self.bn_3 = nn.BatchNorm1d(4096)
        self.softplus_3 = nn.Softplus()
        # output layer
        self.g_4 = nn.Linear(4096, num_classes)
        
    def forward(self, x, y, f_x, t):
        # t = t * 2 / self.T - 1 # NOTE: experiment with "embedding" between -1 and 1
        t = t / self.T
        # encoding input image
        enc_x = self.encoder(x)
        # l_1
        out = torch.cat((y, f_x), dim=1)   # concatenation
        out = self.g_1y(out)
        out = out * self.g_1b(t)
        out = self.bn_1(out)
        out = self.softplus_1(out)
        # integrating image embedding
        out = out * enc_x
        # l_2
        out = self.g_2y(out)
        out = out * self.g_2b(t)
        out = self.bn_2(out)
        out = self.softplus_2(out)
        # l_3
        out = self.g_3y(out)
        out = out * self.g_3b(t)
        out = self.bn_3(out)
        out = self.softplus_3(out)
        # l_4 and return result
        return self.g_4(out)

    def fit(self,
            train_dataloader,
            batch_size=256,
            learning_rate=1e-3,
            epochs=1000,
            T=1000):
        # TODO: Include exponentially weighted moving averages
        # on model paramteres with decay of 0.9999 later.
        
        # Load model to device.
        self.to(DEVICE)

        # Build ADAM optimizer and Loss function
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 40) ## taken from implementation
        
        # Setup finished, enter train mode.
        self.train()

        # Main train loop
        for epoch in range(epochs): # TODO: replace with until convergence
                                    # condition / patience criterum
            # Train for one epoch.
            lossTotal = 0

            # Draw batch.
            for i, batch in enumerate(train_dataloader,0):
                x, y_0 = batch
                x, y_0 = x.to(DEVICE), y_0.to(DEVICE)

                # Zero gradients for every batch.
                optim.zero_grad()

                # Get mean estimations.
                f_x = self.f(x)
                x = torch.flatten(x, 1)

                # Draw t uniformely in {1, ..., T}.
                t = np.random.randint(1, T+1)

                # Draw noises from standard normal distribution.
                eps = torch.normal(0, 1, y_0.shape)
                eps = eps.to(DEVICE)

                # Compute noise estimation loss L_epsilon.
                y_arg = torch.sqrt(self.alpha_bar[t-1]) * y_0
                y_arg += torch.sqrt(1 - self.alpha_bar[t-1]) * eps
                y_arg += (1 - torch.sqrt(self.alpha_bar[t-1])) * f_x
                ##y_arg = F.softmax(y_arg, dim = 1)
                t = torch.tensor([t])
                x, y_arg, f_x, t = x.to(DEVICE), y_arg.to(DEVICE), f_x.to(DEVICE), t.to(DEVICE)
                eps_theta = self.forward(x, y_arg, f_x, t)

                # Perform optimization step.
                L_eps = loss_fn(eps_theta, eps)
                lossTotal += L_eps 
                L_eps.backward()
                optim.step()
                #print(f'Batch {i}/{len(train_dataloader)}, Loss= {L_eps}') ## TODO: temporärer output später entfernen
            scheduler.step()

            # print(f'Epoch {epoch+1:>3d}: loss = {lossTotal / len(batches):>4f}, real noise: {eps[0][0]: >9f}, estimated noise: {eps_theta[0][0]:>9f}, last t = {t:>4d}')        
            if((epoch+1) % 128 == 0 or epoch == 0):
                print(f'Epoch {epoch+1:>3d}: loss = {lossTotal/len(train_dataloader):>4f}')
        print(f'Final loss for the noise estimator: {lossTotal/len(train_dataloader):>4f}')

    def infer(self, x):
        # Set to evaluation mode.
        self.eval()
        self.to(DEVICE)
        self.f.to(DEVICE)

        with torch.no_grad():
            # Calculate y_T for as starting point.
            f_x = self.f(x)
            y_t = torch.normal(f_x, 1)
            x = torch.flatten(x, 1)
            
            # Iterating over all timesteps {T-1, ..., 0}.
            for t in range(self.T-1, -1, -1):

                # Calculating y_hat_0 (here just y_0).
                t_arg = torch.Tensor([[t+1]])
                x, y_t, f_x, t_arg = x.to(DEVICE), y_t.to(DEVICE), f_x.to(DEVICE), t_arg.to(DEVICE)
                y_0 = self.forward(x, y_t, f_x, t_arg)
                y_0 *= -torch.sqrt(1 - self.alpha_bar[t])
                y_0 -= (1 - np.sqrt(self.alpha_bar[t])) * f_x
                y_0 += y_t
                y_0 /= torch.sqrt(self.alpha_bar[t])
                
                # Calculating y_{t-1}.
                if t == 0:
                    y_t = y_0
                else:
                    # Draw z.
                    z = torch.normal(0, 1, y_0.shape)
                    z = z.to(DEVICE)

                    # Gamma terms.
                    divisor = 1 - self.alpha_bar[t]
                    gamma_0 = self.beta[t] * torch.sqrt(self.alpha_bar[t-1]) / divisor
                    gamma_1 = (1 - self.alpha_bar[t -1]) * torch.sqrt(1-self.beta[t]) / divisor
                    gamma_2 = (torch.sqrt(self.alpha_bar[t]) - 1) * (torch.sqrt(1-self.beta[t]) + torch.sqrt(self.alpha_bar[t-1])) / divisor
                    gamma_2 += 1

                    # Beta tilde.
                    beta_tilde_sqrt = torch.sqrt((1 - self.alpha_bar[t-1]) * self.beta[t] / divisor)
                    beta_tilde_sqrt = beta_tilde_sqrt.to(DEVICE)

                    # Combining everything to y_(t-1).
                    if(t % 100 == 0):
                        print(f'Step {1000-t}') ## TODO: temporärer output später entfernen
                    y_t = gamma_1 * y_t
                    y_t += gamma_0 * y_0
                    y_t += gamma_2 * f_x
                    y_t += beta_tilde_sqrt * z
        return F.softmax(y_t, dim = 1)





if __name__ == '__main__':
    # Print if cpu or cuda cores will be used
    print(f"Using {DEVICE} device")

    # Build pre-trained mean estimator.
    M_me = 10240
    train_size_me = M_me * 4 // 5
    test_size_me = M_me - train_size_me
    X_me, y_me = ts.sample_linear_toy_data(M_me, SEED)
    dataset_me = TensorDataset(X_me, y_me)
    train_dataset_me, test_dataset_me = random_split(dataset_me, [train_size_me, test_size_me])
    
    epochs_me = 1000
    loss_fn_me = nn.MSELoss()
    
    # load or train mean estimator
    try:
        mean_estimator = torch.load("mean_est.pth")
    except FileNotFoundError as e:
        print(f"File 'mean_est.pth' not found. Creating and fitting RegressionMeanEstimator ...")
        
        mean_estimator = RegressionMeanEstimator(1)
        optimizer_me = torch.optim.Adam(mean_estimator.parameters())
        mean_estimator.fit(train_dataset_me, optimizer_me)
        torch.save(mean_estimator, 'mean_est.pth')

    # Generate training and test data.
    M = 10240
    train_size = M * 4 // 5
    test_size = M - train_size
    X, y = ts.sample_linear_toy_data(M, SEED+1)
    dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # load or train noise estimator
    try:
        noise_estimator = torch.load("noise_est.pth")
    except FileNotFoundError as e:
        print(f"File 'noise_est.pth' not found. Creating and fitting RegressionNoiseEstimator ...")
        
        noise_estimator = RegressionNoiseEstimator(1, 1, mean_estimator)
        noise_estimator.fit(train_dataset)
        torch.save(noise_estimator, 'noise_est.pth')

    ########################################################
    #                                                      #
    #                 testing inference                    #
    #                                                      #
    ########################################################
    test_infer, y_truth = ts.sample_linear_toy_data(5000)
    labels = noise_estimator.infer(test_infer)
    means = mean_estimator(test_infer).detach().numpy()
    
    import matplotlib.pyplot as plt
    plt.scatter(test_infer, y_truth,s = 5, label = 'True Dist')
    plt.scatter(test_infer,labels, s = 5, label = 'Diffusion')
    plt.scatter(test_infer, means,s = 1, label = 'Mean Est', color = 'Aqua')
    plt.grid(True)
    plt.legend()
    plt.show()
