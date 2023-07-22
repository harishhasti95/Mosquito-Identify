import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms, utils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class mosquitoDatasetClassification(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mosquito_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        self.encoding = {value: index for index, value in enumerate(['albopictus', 'culex', 'anopheles', 'culiseta', 'japonicus/koreicus', 'aegypti'])}
    
    def __len__(self):
        return len(self.mosquito_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # print(self.mosquito_frame['img_fName'].loc[idx])
        img_name = os.path.join(self.root_dir,
                                self.mosquito_frame['img_fName'].loc[idx])

        image = Image.open(img_name)
        # print(type(image))
        # image = ToTensor()(image).unsqueeze(0)
        # image = Variable(image)
        label = self.mosquito_frame['class_label'].loc[idx]



        label_encoded = torch.zeros(6)
        label_encoded[self.encoding[label]] = 1
        
        sample = {'image': image, 'label': label_encoded}

        if self.transform:
            sample['image'] = self.transform(sample['image'])


        return sample