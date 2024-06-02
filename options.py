
class Options():
    def __init__(self):
        super().__init__()
        self.Seed = 1234
        self.Epoch = 400
        self.Learning_Rate = 2e-4
        self.Batch_Size_Train = 18
        self.Batch_Size_Val = 18
        self.Patch_Size_Train = 128
        self.Patch_Size_Val = 128

        # 训练集路径
        self.Input_Path_Train = '/storage/public/home/2022024024/dataset/TITS-all-in-one/train_/input'
        self.Target_Path_Train = '/storage/public/home/2022024024/dataset/TITS-all-in-one/train_/target'

        # 验证集路径
        self.Input_Path_Val = '/storage/public/home/2022024024/dataset/TITS-all-in-one/test/RainDS-syn/input'
        self.Target_Path_Val = '/storage/public/home/2022024024/dataset/TITS-all-in-one/test/RainDS-syn/target'

        # 测试集路径
        self.Dataset_Names = ['FogCityscapes',
                              'RainCityscapes',
                              'RSCityscapes',
                              'SnowTrafficData',
                              'LowLightTrafficData',
                              'RainDS-syn']
        self.Path_Test = '/storage/public/home/2022024024/dataset/TITS-all-in-one/test'

        self.MODEL_RESUME_PATH = './model_best.pth'

        self.Num_Works = 4
        self.CUDA_USE = True