import torch

def dataloader(dataset, batch_size=4, shuffle=True, num_workers=2):
    datasetLoader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return datasetLoader
