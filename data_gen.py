"""
Methods for making 'real' data from a choice of several
1D functions
"""
import random
import torch
from torch.utils.data import Dataset

def power_x(x: torch.Tensor, power: float):
    return x ** power

def sin_x(x: torch.Tensor):
    return torch.sin(x)

def cos_x(x: torch.Tensor):
    return torch.cos(x)

def exp_x(x: torch.Tensor):
    return torch.exp(x) 

METHODS = {
    'power': power_x,
    'sin': sin_x,
    'cos': cos_x,
    'exp': exp_x
}

def generate_points(n: int = 100, func: str = 'power', **kwargs):
    """
    Generate random data points according to some specified function

    Parameters
    ----------
    n: int, default=100
        Number of points to generate
    func: str, default='power'
        Mathematical function to use to generate points. Accepted values are:
        ['power', 'sin', 'cos', 'exp']

    Returns
    -------
    data: torch.Tensor
        Generated data points
    """
    # generate points in range [0, 1]
    x = torch.rand(n) 
    # find y-values
    y = METHODS[func](x, **kwargs) 
    # reshape so each point is a different sample
    x = torch.reshape(x, [-1, 1])
    y = torch.reshape(y, [-1, 1])
    # stack together to make sample
    data = torch.cat([x, y], 1)
    return data

class TrainingSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

def generate_dataset(n_points: int):
    data = torch.Tensor()
    target = torch.Tensor()
    for i in range(n_points):
        real = random.choice([True, False])
        if real:
            x = generate_points(1, power=2)
            y = torch.Tensor([1])
        else:
            x = generate_points(1, 'cos')
            y = torch.Tensor([0])
        data = torch.cat([data, x])
        target = torch.cat([target, y])
    target = torch.reshape(target, [-1, 1])
    dataset = TrainingSet(data, target)
    return dataset

if __name__ == '__main__':

    print(generate_data(10, 'power', power=2))