import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import scipy.misc


class DriveData(Dataset):
    __xs = []
    __ys = []

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data (And load to memory)
        with open(folder_dataset + "data.txt") as f:
            for line in f:
                # Image path
                self.__xs.append(folder_dataset + line.split()[0])
                # Steering wheel label
                self.__ys.append(np.float32(line.split()[1]))

    def addFolder(self, folder_dataset):
        # Open and load text file including the whole training data
        with open(folder_dataset + "data.txt") as f:
            for line in f:
                # Image path
                self.__xs.append(folder_dataset + line.split()[0])
                # Steering wheel label
                self.__ys.append(np.float32(line.split()[1]))

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = scipy.misc.imread(self.__xs[index], mode="RGB")
        img = (scipy.misc.imresize(img[126:226], [66, 200]) / 255.0)

        # Do Transformations on the image
        if self.transform is not None:
            img = self.transform(img)

        # #img = torch.from_numpy(np.asarray(img)) (We have a transform to do the job)
        # Convert label to torch tensors
        label = self.__ys[index]
        #label = np.float32(-1.0)
        #label = torch.from_numpy(np.asarray(label).reshape([1, 1]))
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)