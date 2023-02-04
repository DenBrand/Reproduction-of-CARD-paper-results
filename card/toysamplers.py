import numpy as np
import torch
from typing import Union

def sample_linear_toy_data(sample_amount: int, seed: Union[int, None]=None):
    """
    Generates data for the Linear toy example according to the formula y = 3x + 3 + epsilon
    x ~ U(-5, 5), epsilon ~ N(0, 2^2)
    """
    if seed:
        torch.manual_seed(seed)

    x = torch.rand(sample_amount, 1) * 10 - 5
    noise = torch.normal(0, 2, size=(sample_amount, 1))
    y = 3 * x + 3 + noise
    return x, y

def sample_quadratic_toy_data(sample_amount: int, seed: Union[int, None]=None):
    """
    Generates data for the Quadratic toy example according to the formula y = 3x^2 + 2x + 1 + epsilon
    x ~ U(-5, 5), epsilon ~ N(0, 2^2)
    """
    if seed:
        torch.manual_seed(seed)
 
    x = torch.rand(sample_amount, 1) * 10 - 5
    noise = torch.normal(0, 2, size=(sample_amount, 1))
    y = 3 * x**2 + 2*x + 1 + noise

    return x, y

def sample_loglog_linear_toy_data(sample_amount: int, seed: Union[int, None]=None):
    """
    Generates data for the Log-Log Linear toy example according to the formula y = exp( log(x) + epsilon)
    x ~ U(0, 10), epsilon ~ N(0, 0.15^2)
    """
    if seed:
        torch.manual_seed(seed)

    x = torch.rand(sample_amount, 1) * 10
    noise = torch.normal(0, 0.15, size=(sample_amount, 1))
    y = torch.exp(torch.log(x) + noise)

    return x, y

def sample_loglog_cubic_toy_data(sample_amount: int, seed: Union[int, None]=None):
    """
    Generates data for the Log-Log Cubic toy example according to the formula y = exp( 3 * log(x) + epsilon)
    x ~ U(0, 10), epsilon ~ N(0, 0.15^2)
    """
    if seed:
        torch.manual_seed(seed)

    x = torch.rand(sample_amount, 1) * 10
    noise = torch.normal(0, 0.15, size=(sample_amount, 1))
    y = torch.exp(3*torch.log(x) + noise)

    return x, y

def sample_sinusoidal_toy_data(sample_amount: int, seed: Union[int, None]=None):
    """
    Generates data for the Sinusoidal toy example according to the formula y = x + 0.3 sin(2pi x) + epsilon
    x ~ U(0, 1), epsilon ~ N(0, 0.08^2)
    """
    if seed:
        torch.manual_seed(seed)

    x = torch.rand(sample_amount, 1)
    noise = torch.normal(0, 0.08, size=(sample_amount, 1))
    y = x + 0.3 * torch.sin(2*np.pi*x) + noise

    return x, y


def sample_inverse_sinusoidal_toy_data(sample_amount: int, seed: Union[int, None]=None):
    """
    Generates data for the Inverse Sinusoidal toy example according to the formula x = y + 0.3 sin(2pi y) + epsilon
    y ~ U(0, 1), epsilon ~ N(0, 0.08^2). This is a flipped version of the Sinusoidal toy example
    """
    if seed:
        torch.manual_seed(seed)

    y = torch.rand(sample_amount, 1)
    noise = torch.normal(0, 0.08, size=(sample_amount, 1))
    x = y + 0.3 * torch.sin(2*np.pi*y) + noise

    return x, y

def sample_8_gaussians_toy_data(sample_amount: int, seed: Union[int, None]=None):
    """
    Generates data for the 8 Gaussians toy example with gaussian means of (+- sqrt(2), 0); (0, +- sqrt(2)) and (+-1, +-1) 
    epsilon in both axes is ~ N(0, 0.1^2)
    """
    if seed:
        torch.manual_seed(seed)

    x = torch.randint(-1, 3, (sample_amount, 1)).float()
    y = 2 - torch.abs(x)
    y = torch.sqrt(y)
    x[x == 2] = np.sqrt(2)

    sign = torch.randint(0, 2, (sample_amount, 1)) * 2 - 1
    x = x * sign
    sign = torch.randint(0, 2, (sample_amount, 1)) * 2 - 1
    y = y * sign

    noise = torch.normal(0, 0.1, size=(sample_amount, 1))
    x = x + noise
    noise = torch.normal(0, 0.1, size=(sample_amount, 1))
    y = y + noise

    return x, y

def sample_full_circle_toy_data(sample_amount: int, seed: Union[int, None]=None):
    """
    Generates data for the Full Circle toy example where x = 10 cos(2pi*z) + epsilon and y = 10 sin(2pi*z) + epsilon
    z ~ U(0, 1), epsilon ~ N(0, 0.5^2)
    """
    if seed:
        torch.manual_seed(seed)

    x = torch.rand(sample_amount, 1)
    noise = torch.normal(0, 0.5, size=(sample_amount, 1))
    y = 10 * torch.sin(2 * x * np.pi) + noise
    noise = torch.normal(0, 0.5, size=(sample_amount, 1))
    x = 10 * torch.cos(2 * x * np.pi) + noise

    return x, y

def sample_toy_data_by_index(sample_amount: int, seed: Union[int, None]=None, index = 1):
    """
    Generates toy data depending on the index, used for debugging for now
    Index 1 = Linear Toy Data
    Index 2 = Quadratic Data
    Index 3 = LogLog Linear Data
    Index 4 = Loglog Cubic
    Index 5 = Sinusoidal
    Index 6 = Inverse Sinusoidal
    Index 7 = 8 Gaussians
    Index 8 = Full Circle
    """
    assert index > 0 and index < 9, 'index parameter is expected to be in [1, 8]'

    if index == 1:
        x,y = sample_linear_toy_data(sample_amount, seed)
    elif index == 2:
        x,y = sample_quadratic_toy_data(sample_amount, seed)
    elif index == 3:
        x,y = sample_loglog_linear_toy_data(sample_amount, seed)
    elif index == 4:
        x,y = sample_loglog_cubic_toy_data(sample_amount, seed)
    elif index == 5:
        x,y = sample_sinusoidal_toy_data(sample_amount, seed)
    elif index == 6:
        x,y = sample_inverse_sinusoidal_toy_data(sample_amount, seed)
    elif index == 7:
        x,y = sample_8_gaussians_toy_data(sample_amount, seed)
    else:
        x,y = sample_full_circle_toy_data(sample_amount, seed)
    return x,y

def get_correct_mean(x, index):
    assert index > 0 and index < 9, 'index parameter is expected to be in [1, 8]'

    if index == 1:
        return 3 * x + 3
    elif index == 2:
        return 3 * x**2 + 2*x + 1
    elif index == 3:
        return torch.exp(torch.log(x))
    elif index == 4:
        return  torch.exp(3*torch.log(x))
    elif index == 5:
        return x + 0.3 * torch.sin(2*np.pi*x)
    else:
        return 0

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    for index in range(1,9):
        x,y = sample_toy_data_by_index(1000, index = index)
        x.reshape((1000,))
        plt.scatter(x,y, s = 5)
        plt.grid(True)
        plt.show()