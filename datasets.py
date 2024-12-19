import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf


class MyTrainDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=256):
        super(MyTrainDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.to_tensor(inputImage)
        targetImage = ttf.to_tensor(targetImage)

        hh, ww = targetImage.shape[1], targetImage.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        aug = random.randint(0, 8)

        input_ = inputImage[:, rr:rr+ps, cc:cc+ps]
        target = targetImage[:, rr:rr+ps, cc:cc+ps]

        if aug == 1:
            input_, target = input_.flip(1), target.flip(1)
        elif aug == 2:
            input_, target = input_.flip(2), target.flip(2)
        elif aug == 3:
            input_, target = torch.rot90(input_, dims=(1, 2)), torch.rot90(target, dims=(1, 2))
        elif aug == 4:
            input_, target = torch.rot90(input_, dims=(1, 2), k=2), torch.rot90(target, dims=(1, 2), k=2)
        elif aug == 5:
            input_, target = torch.rot90(input_, dims=(1, 2), k=3), torch.rot90(target, dims=(1, 2), k=3)
        elif aug == 6:
            input_, target = torch.rot90(input_.flip(1), dims=(1, 2)), torch.rot90(target.flip(1), dims=(1, 2))
        elif aug == 7:
            input_, target = torch.rot90(input_.flip(2), dims=(1, 2)), torch.rot90(target.flip(2), dims=(1, 2))

        return input_, target


class MyValueDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=64):
        super(MyValueDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.center_crop(inputImage, (ps, ps))
        targetImage = ttf.center_crop(targetImage, (ps, ps))

        input_ = ttf.to_tensor(inputImage)
        target = ttf.to_tensor(targetImage)

        return input_, target


class MyTestDataSet(Dataset):
    def __init__(self, inputPath, targetPath=None):
        super(MyTestDataSet, self).__init__()

        self.inputPath = inputPath
        self.inputImages = os.listdir(inputPath)

        self.targetPath = targetPath
        if self.targetPath is not None:
            self.targetPath = targetPath
            self.targetImages = os.listdir(targetPath)

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):

        index = index % len(self.inputImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        input_ = ttf.to_tensor(inputImage)

        if self.targetPath is not None:
            targetImagePath = os.path.join(self.targetPath, self.inputImages[index])
            targetImage = Image.open(targetImagePath).convert('RGB')
            target = ttf.to_tensor(targetImage)

            return input_, target, self.inputImages[index]
        else:
            return input_, self.inputImages[index]