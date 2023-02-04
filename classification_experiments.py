import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from card.architectures import ClassificationPretrainedModel, ClassificationNoiseEstimator
import torch.nn.functional as F
from os.path import join

import sys
import card.architectures

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
DATASETS_DIRECTORY = 'datasets/classification/'
MODEL_DIRECTORY = join('models', 'classification')


def load_Cifar(batch_size =256): ##TODO normalize the data
    ## temporary change https to http so it runs on my pc
    torchvision.datasets.CIFAR10.url = torchvision.datasets.CIFAR10.url.replace('https://', 'http://')
    cifar_directory = join(DATASETS_DIRECTORY, 'Cifar10')

    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    target_transform=torchvision.transforms.Compose([
                                 lambda x:torch.tensor([x]),
                                 lambda x:F.one_hot(x,10),
                                 lambda x:torch.squeeze(x)])

    trainset = torchvision.datasets.CIFAR10(root = cifar_directory, 
                                            train = True, 
                                            download = True, 
                                            transform = transform, 
                                            target_transform = target_transform)
    trainloader = DataLoader(trainset, 
                            batch_size = batch_size,
                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root = cifar_directory, 
                                           train = False,
                                           download = True,
                                           transform = transform, 
                                           target_transform = target_transform)
    testloader = DataLoader(testset, 
                            batch_size = batch_size,
                            shuffle = False)

    return trainloader, testloader


if __name__ == "__main__":

    trainloader, testloader = load_Cifar()

    # Load or fit the pretrained model
    pretrained_model: ClassificationPretrainedModel = None
    mean_estimator_path = join(MODEL_DIRECTORY, f'Classification_pretrained_model.pth')
    try:
        pretrained_model = torch.load(mean_estimator_path)
    except FileNotFoundError as e:
        print(f"File {mean_estimator_path} not found. Creating and fitting RegressionMeanEstimator...")
        pretrained_model = ClassificationPretrainedModel()
        optimizer = torch.optim.Adam(pretrained_model.parameters())
        pretrained_model.fit(trainloader, optimizer)
        # for i in range(10):
        #     print(f'Epoch {i}:')
        #     pretrained_model.train_one_epoch(trainloader, optimizer, verbose = False)
    torch.save(pretrained_model, mean_estimator_path)
    pretrained_model.test(testloader, verbose = False)

    # Load or fit the Noise estimator
    card_model: ClassificationNoiseEstimator = None
    card_model_path = join(MODEL_DIRECTORY, f'Classification_Card_Model.pth')
    try:
        card_model = torch.load(card_model_path)
    except FileNotFoundError as e:
        print(f"File {card_model_path} not found. Creating and fitting Card Model...")
        card_model = ClassificationNoiseEstimator(32*32*3, 10, pretrained_model) ## TODO inputs verallgemeinern
        card_model.fit(trainloader, epochs = 1)  ##TODO Epochen erhöhen
        torch.save(card_model, card_model_path)


    ## Temporary test #TODO remove later
    for i, batch in enumerate(testloader,0):
        x, y = batch
        print(x.shape)
        test = card_model.infer(x)
        print(test)
        print(y)
        break

