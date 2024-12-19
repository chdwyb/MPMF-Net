import os
import sys
import torch
import random
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import MPMFNet
from options import Options
from perceptual import LossNetwork
from utils import torchPSNR, CharbonnierLoss
from datasets import MyTrainDataSet, MyValueDataSet


if __name__ == '__main__':

    opt = Options()
    cudnn.benchmark = True

    random.seed(opt.Seed)
    torch.manual_seed(opt.Seed)
    torch.cuda.manual_seed(opt.Seed)
    torch.manual_seed(opt.Seed)

    EPOCH = opt.Epoch
    BATCH_SIZE_TRAIN = opt.Batch_Size_Train
    BATCH_SIZE_VAL = opt.Batch_Size_Val
    PATCH_SIZE_TRAIN = opt.Patch_Size_Train
    PATCH_SIZE_VAL = opt.Patch_Size_Val
    LEARNING_RATE = opt.Learning_Rate

    inputPathTrain = opt.Input_Path_Train
    targetPathTrain = opt.Target_Path_Train
    inputPathVal = opt.Input_Path_Val
    targetPathVal = opt.Target_Path_Val

    myNet = MPMFNet()
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    myNet = nn.DataParallel(myNet, device_ids)
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.cuda()
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    optimizer = optim.Adam(myNet.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, EPOCH, eta_min=1e-7)

    datasetTrain = MyTrainDataSet(inputPathTrain, targetPathTrain, patch_size=PATCH_SIZE_TRAIN)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    datasetValue = MyValueDataSet(inputPathVal, targetPathVal, patch_size=PATCH_SIZE_VAL)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=BATCH_SIZE_VAL, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    if os.path.exists(opt.MODEL_RESUME_PATH):
        if opt.CUDA_USE:
            myNet.load_state_dict(torch.load(opt.MODEL_RESUME_PATH))
        else:
            myNet.load_state_dict(torch.load(opt.MODEL_RESUME_PATH, map_location=torch.device('cpu')))

    best_psnr = 0
    psnr_val_rgb = 0
    scalar = torch.cuda.amp.GradScaler()
    for epoch in range(EPOCH):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        for index, (x, y) in enumerate(iters, 0):

            myNet.zero_grad()
            optimizer.zero_grad()

            if opt.CUDA_USE:
                input_train, target = Variable(x).cuda(), Variable(y).cuda()
            else:
                input_train, target = Variable(x), Variable(y)

            with torch.cuda.amp.autocast(True):
                restored = myNet(input_train)

                loss_inr = F.l1_loss(restored[1], F.interpolate(target, scale_factor=0.25))
                loss_l1 = F.l1_loss(restored[0], target)
                loss_perc = loss_network(restored[0], target)
                loss = loss_l1 + 0.04 * loss_perc + 0.1 * loss_inr

                scalar.scale(loss).backward()
                scalar.step(optimizer)
                scalar.update()

            iters.set_description('Training !!!  Epoch %d / %d' % (epoch+1, EPOCH))

        if epoch % 3 == 0:
            myNet.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_, target_value = (x.cuda(), y.cuda()) if opt.CUDA_USE else (x, y)
                with torch.no_grad():
                    output_value = myNet(input_)
                for output_value, target_value in zip(output_value[0], target_value):
                    psnr_val_rgb.append(torchPSNR(output_value, target_value))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb >= best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(myNet.state_dict(), './model_best.pth')

        if (epoch + 1) % 20 == 0:
            torch.save(myNet.state_dict(), f'./model_{epoch + 1}.pth')
        scheduler.step(epoch)
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Epoch Loss:  {:.6f}, Current PSNR:  {:.3f}, Best PSNR:  {:.3f}.".format(
                epoch + 1, epochLoss, psnr_val_rgb, best_psnr))
        print('-------------------------------------------------------------------------------------------------------')






