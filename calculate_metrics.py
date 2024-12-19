import os
import cv2
import torch
import lpips
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as ttf
from concurrent.futures import ThreadPoolExecutor


class MyDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain):
        super(MyDataSet, self).__init__()
        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)
        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):
        index = index % len(self.targetImages)
        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')
        targetImagePath = os.path.join(self.targetPath, self.inputImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')
        input_ = ttf.to_tensor(inputImage)
        target = ttf.to_tensor(targetImage)
        return input_, target


def cal_psnr(image_pair):
    input_image_path, target_image_path = image_pair
    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)

    if input_image is None or target_image is None:
        raise ValueError(f"Error reading images: {input_image_path}, {target_image_path}")

    return calculate_psnr(target_image, input_image, test_y_channel=True)


def cal_ssim(image_pair):
    input_image_path, target_image_path = image_pair
    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)

    if input_image is None or target_image is None:
        raise ValueError(f"Error reading images: {input_image_path}, {target_image_path}")

    return calculate_ssim(target_image, input_image, test_y_channel=True)


def get_image_pairs(input_dir, target_dir):
    image_names = os.listdir(input_dir)
    image_pairs = [
        (os.path.join(input_dir, name), os.path.join(target_dir, name))
        for name in image_names
        if os.path.isfile(os.path.join(input_dir, name)) and os.path.isfile(os.path.join(target_dir, name))
    ]
    return image_pairs


def calculate_average_psnr(input_dir, target_dir):
    image_pairs = get_image_pairs(input_dir, target_dir)

    with ThreadPoolExecutor() as executor:
        psnr_values = list(executor.map(cal_psnr, image_pairs))

    return np.mean(psnr_values) if psnr_values else 0


def calculate_average_ssim(input_dir, target_dir):
    image_pairs = get_image_pairs(input_dir, target_dir)

    with ThreadPoolExecutor() as executor:
        ssim_values = list(executor.map(cal_ssim, image_pairs))

    return np.mean(ssim_values) if ssim_values else 0


def calculate_average_lpips(input_dir, target_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calculate_lpips = lpips.LPIPS(net='alex', verbose=False).to(device)

    datasetTest = MyDataSet(input_dir, target_dir)
    testLoader = DataLoader(dataset=datasetTest)
    lpips_total = 0
    for index, (x, y) in enumerate(testLoader):
        result, target = x.to(device), y.to(device)
        lpips_total += calculate_lpips(result * 2 - 1, target * 2 - 1).squeeze().item()
    return lpips_total / index


datasets = [
    'FoggyCityscapes',
    'RainCityscapes',
    'RSCityscapes',
    'SnowTrafficData',
    'LowlightTrafficData',
    'RainDS-syn'
    ]

average_psnr, average_ssim, average_lpips = 0, 0, 0
for dataset in datasets:
    # 数据路径
    path_result = 'G://traffic-all-in-one/test/result' + dataset
    path_target = 'G://traffic-all-in-one/test/target' + dataset
    average_psnr = calculate_average_psnr(path_result, path_target)
    average_ssim = calculate_average_ssim(path_result, path_target)
    average_lpips = calculate_average_lpips(path_result, path_target)

    print(f'{dataset}: PSNR: {average_psnr}, SSIM: {average_ssim}, LPIPS: {average_lpips}')