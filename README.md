###  Multi-axis Prompt and Multi-dimension Fusion Network for All-in-one Weather-degraded Image Restoration

**Abstract**: Existing approaches aiming to remove adverse weather degradations compromise the image quality and incur the long processing time. To this end, we introduce a multi-axis prompt and multi-dimension fusion network (MPMF-Net). Specifically, we develop a multi-axis prompts learning block (MPLB), which learns the prompts along three separate axis planes, requiring fewer parameters and achieving superior performance. Moreover, we present a multi-dimension feature interaction block (MFIB), which optimizes intra-scale feature fusion by segregating features along height, width and channel dimensions. This strategy enables more accurate mutual attention and adaptive weight determination. Additionally, we propose the coarse-scale degradation-free implicit neural representations (CDINR) to normalize the degradation levels of different weather conditions. Extensive experiments demonstrate the significant improvements of our model over the recent well-performing approaches in both reconstruction fidelity and inference time.

### News 🚀🚀🚀

- **December 10, 2024**: Our work is accepted by AAAI 2025.
- **June 2, 2024**: We released the code and pre-trained models.

### Requirements

```python
python 3.11.5
torch 2.0.1
torchvision 0.15.2
tqdm 4.66.1
pillow 9.3.0
numpy 1.24.1
lpips 0.1.4
opencv-python 4.8.1.78
```

### Datasets

We provided the involved training and testing datasets in the following links.

| Dataset                                 | Link                                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| all-in-one dataset (AWTD) for training  | [[Baidu Netdisk](https://pan.baidu.com/s/1PG7rRsSUi_j7X41L92ixcA), code: awtd] |
| FoggyCityscapes dataset for testing     | [[Baidu Netdisk](https://pan.baidu.com/s/1oCdB4I5LyQTluvAZXTSFuA), code: awtd] |
| RainCityscapes dataset for testing      | [[Baidu Netdisk](https://pan.baidu.com/s/128nN7fBJXVn8FpxCLq6VeQ), code: awtd] |
| RSCityscapes dataset for testing        | [[Baidu Netdisk](https://pan.baidu.com/s/1sPxguAQ3auftmvz2t_pu9w), code: awtd] |
| SnowTrafficData dataset for testing     | [[Baidu Netdisk](https://pan.baidu.com/s/1zB1ulVY1rU1Z2BOHXSN0cg), code: awtd] |
| LowLightTrafficData dataset for testing | [[Baidu Netdisk](https://pan.baidu.com/s/1aD24M_A9yRoe5bMMb-pIuA), code: awtd] |
| RainDS-syn dataset for testing          | [[Baidu Netdisk](https://pan.baidu.com/s/1I2ydU550vmOmOrfQBUNn1w), code: awtd] |

Please prepare these datasets following the form below.

```pyth
|--train
    |--input  
        |--image 1
        : 
        |--image n
    |--target  
        |--image 1
        :
        |--image n
|--test 
    |--FoggyCityscapes
        |--input  
            |--image 1
            :  
            |--image m
    	|--target  
            |--image 1
            : 
            |--image m
	|--RainCityscapes
	|--RSCityscapes
	|--SnowTrafficData
	|--LowLightTrafficData
	|--RainDS-syn
```

### Train

We provide the training code in the `train.py` file for training our proposed MPMF-Net, you may utilize the following example usage to conduct training. All hyper-parameters and paths can be modified in `options.py`.

```python
python train.py
```

### Test

To evaluate our MPMF-Net, you may download our trained model from [[Baidu Netdisk](https://pan.baidu.com/s/1ygiUO1nFarWDfCFQsoFCuQ), code: awtd], and utilize the following example usage to conduct experiments. 

```python
python test.py
```

### Result

For any possible needs, we also provide the results all relevant methods in [[Baidu Netdisk](https://pan.baidu.com/s/1kBRwdC-gobaovUAMf8KSUg), code: awtd].

### Contact us

If I have any questions regarding our work, please feel free to contact us at [wyb@chd.edu.cn](mailto:wyb@chd.edu.cn).