from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from datasets import load_dataset

import torch.utils.data as data_utils
import os


class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True) 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.dataframe.iloc[idx].to_dict() 
        return sample
    
def get_paradetox_train_and_val_datasets():
    path = "/home/ubuntu/20thao.nt/TST/count-style-transfer/datasets/paradetox"
    dataset = pd.read_csv(os.path.join(path,'paradetox_test.csv'))
    # dataset = load_dataset("/home/ubuntu/20thao.nt/TST/count-style-transfer/datasets/paradetox", "en-US", split="train")
    N = len(dataset)

    train_size = int(0.8* N)
    test_size = N - train_size

    generator1 = torch.Generator().manual_seed(42)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],generator=generator1)
    train_dataset, val_dataset = torch.utils.data.random_split(PandasDataset(dataset), [train_size, test_size], generator=generator1)
    print("Length of train dataset: ", len(train_dataset))
    print("Length of val dataset: ", len(val_dataset))
    # print(train_dataset[:5])
    return train_dataset, val_dataset

def get_paradetox_train_and_val_loaders(batch_size=8):
    train_dataset, val_dataset = get_paradetox_train_and_val_datasets()

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader


def get_APPDIA_train_and_val_loaders(path='/home/ubuntu/20thao.nt/TST/count-style-transfer/datasets/APPDIA',batch_size=8):

    # train_dataset = load_dataset(path,split="train")
    # val_dataset = load_dataset(path,split="validation")
    train_dataset = pd.read_csv(os.path.join(path,'train.tsv'), sep = '\t')
    val_dataset = pd.read_csv(os.path.join(path,'validation.tsv'), sep = '\t')

    train_dataset = PandasDataset(train_dataset)
    val_dataset = PandasDataset(val_dataset)
    # print(train_dataset)
    print("Length of train dataset: ", len(train_dataset))
    print("Length of val dataset: ", len(val_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader