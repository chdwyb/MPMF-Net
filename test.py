import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import MPMFNet
from options import Options
from datasets import MyTestDataSet


if __name__ == '__main__':

    opt = Options()

    myNet = MPMFNet()
    myNet = nn.DataParallel(myNet)
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    if opt.CUDA_USE:
        myNet.load_state_dict(torch.load(opt.MODEL_RESUME_PATH))
    else:
        myNet.load_state_dict(torch.load(opt.MODEL_RESUME_PATH, map_location=torch.device('cpu')))
    myNet.eval()

    for dataset in opt.Dataset_Names:
        inputPathTest = os.path.join(opt.Path_Test, dataset, 'input')
        resultPathTest = os.path.join(opt.Path_Test, dataset, 'result')
        os.makedirs(resultPathTest, exist_ok=True)

        datasetTest = MyTestDataSet(inputPathTest)
        testLoader = DataLoader(dataset=datasetTest)

        with torch.no_grad():
            for index, (x, name) in enumerate(tqdm(testLoader, desc=f'{dataset} Testing !!!', file=sys.stdout), 0):
                torch.cuda.empty_cache()
                input_test = x.cuda() if opt.CUDA_USE else x
                output_test, _ = myNet(input_test)
                save_image(output_test, os.path.join(resultPathTest, name[0]))

