import os
import cv2
import lpips
from options import Options
from datasets import MyTestDataSet
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from torch.utils.data import DataLoader


if __name__ == '__main__':

    opt = Options()

    for dataset in opt.Dataset_Names:
        targetPathTest = os.path.join(opt.Path_Test, dataset, 'target')
        resultPathTest = os.path.join(opt.Path_Test, dataset, 'result')

        psnr_total = 0
        ssim_total = 0
        lpips_total = 0

        image_list = os.listdir(resultPathTest)
        L = len(image_list)

        for image_name in image_list:

            result_image_path = os.path.join(resultPathTest, image_name)
            image_result = cv2.imread(result_image_path, cv2.IMREAD_COLOR)
            target_image_path = os.path.join(targetPathTest, image_name)
            image_target = cv2.imread(target_image_path, cv2.IMREAD_COLOR)

            psnr_total += calculate_psnr(image_result, image_target, test_y_channel=True)
            ssim_total += calculate_ssim(image_result, image_target, test_y_channel=True)

        calculate_lpips = lpips.LPIPS(net='alex', verbose=False)
        if opt.CUDA_USE:
            calculate_lpips = calculate_lpips.cuda()

        datasetTest = MyTestDataSet(resultPathTest, targetPathTest)
        testLoader = DataLoader(dataset=datasetTest)

        for index, (x, y, _) in enumerate(testLoader):

            result = x.cuda() if opt.CUDA_USE else x
            target = y.cuda() if opt.CUDA_USE else y
            lpips_total += calculate_lpips(result * 2 - 1, target * 2 - 1).squeeze().item()

        print('{}, PSNR: {:.3f}, SSIM: {:.4f}, LPIPS: {:.4f}'.format(dataset, psnr_total / L, ssim_total / L, lpips_total / L))