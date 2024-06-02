###  Multi-axis Prompting and Multi-dimension Fusion Network for All-in-one Traffic Weather Removal

**Abstract**: Existing approaches aiming to remove adverse weather degradations often compromise the image quality and incur the long processing time, while overlooking complex traffic scenarios. To this end, we introduce a multi-axis prompting and multi-dimension fusion network (MPMF-Net) for all-in-one traffic weather removal. Specifically, we propose the coarse-scale degradation-free implicit neural representations (CDINR) to obtain the non-degraded neural representations, which normalizes the degradation levels of different weather conditions and serves as visual prompts to direct the subsequent reconstruction. Moreover, we develop a multi-axis prompts learning block (MPLB), which learns the prompts along three separate axis planes, requiring fewer parameters while achieving superior prompting learning. Additionally, we present a multi-dimension feature interaction block (MFIB), which optimizes feature fusion by segregating features along height, width and channel dimensions. This strategy enables the multi-dimension mutual attention and adaptive weight determination, leading to more accurate cross-stage intra-scale feature fusion. To facilitate research, we provide a comprehensive dataset comprising diverse traffic weather-degraded image pairs. Extensive experiments demonstrate the superiority of our model over the recent well-performing DyNet, achieving a 0.89 dB performance improvement with only 9.96% of the parameter amount and 17.60% of the inference time.

### News 🚀🚀🚀

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

We provided the involved datasets in the following links.

| Dataset                     | Link                                                         |
| --------------------------- | ------------------------------------------------------------ |
| All-in-one Dataset          | [[Baidu Netdisk](https://pan.baidu.com/s/17D03FKWYkjBOwtmGZKpOyw), code: awtd] |
| FoggyCityscapes Dataset     | [[Baidu Netdisk](https://pan.baidu.com/s/1pUUTZMJa71uO3KnAxFvg0A), code: awtd] |
| RainCityscapes Dataset      | [[Baidu Netdisk](https://pan.baidu.com/s/1qUd3_0RV2Exxjb3mJDMHgw), code: awtd] |
| RSCityscapes Dataset        | [[Baidu Netdisk](https://pan.baidu.com/s/1ok62yPuIa82sscTM_LT1Qg), code: awtd] |
| SnowTrafficData Dataset     | [[Baidu Netdisk](https://pan.baidu.com/s/1aRVhFX5FdgtsoZ__kKG7FA), code: awtd] |
| LowLightTrafficData Dataset | [[Baidu Netdisk](https://pan.baidu.com/s/1s3GG4mlpHIDlBm7BtytKjQ), code: awtd] |
| RainDS-syn Dataset          | [[Baidu Netdisk](https://pan.baidu.com/s/1b6IGBMF1cQMqW9ui1g4Qqw), code: awtd] |

### Pre-trained Models

We released the pre-trained models in the following links.

| Dataset                     | Link                                                         |
| --------------------------- | ------------------------------------------------------------ |
| All-in-one Dataset          | [[Baidu Netdisk](https://pan.baidu.com/s/1gtb_6oMjpEfPuhmkeGqP2g), code: awtd] [[Google Drive](https://drive.google.com/file/d/1Qmfiy2HphVhvaOCtq_GDvOrkPX1kXI3r/view?usp=sharing)] |
| FoggyCityscapes Dataset     | [[Baidu Netdisk](https://pan.baidu.com/s/1D-gplkVfDsJnlsbZb4wlyg), code: awtd] [[Google Drive](https://drive.google.com/file/d/1Ls87lsK3SA3R-MnaKS4LvuB75P8_dzsd/view?usp=sharing)] |
| RainCityscapes Dataset      | [[Baidu Netdisk](https://pan.baidu.com/s/19ekMLLFtw5rXy3xufTqM2Q), code: awtd] [[Google Drive](https://drive.google.com/file/d/1d5ZaaMWIiMmfDvPv6P4MCjUZkyZRkM32/view?usp=sharing)] |
| RSCityscapes Dataset        | [[Baidu Netdisk](https://pan.baidu.com/s/1CbYkYYADcoXDpM3Seob-Cw), code: awtd] [[Google Drive](https://drive.google.com/file/d/1I1ANYR6A9Sxuv-qPZ7E35YC8R7Q0HOps/view?usp=sharing)] |
| SnowTrafficData Dataset     | [[Baidu Netdisk](https://pan.baidu.com/s/1nHp2zKXOFqVRWHxfpaRWvA), code: awtd] [[Google Drive](https://drive.google.com/file/d/1b09NkJ0-lt4G-HIEkXOTXIDqSbCH8skK/view?usp=sharing)] |
| LowLightTrafficData Dataset | [[Baidu Netdisk](https://pan.baidu.com/s/1OFGG5KzNCEODGSKeKl4gbg), code: awtd] [[Google Drive](https://drive.google.com/file/d/1_JCME8VgGCijpYzREfYm9gdjoJDFr5XL/view?usp=sharing)] |
| RainDS-syn Dataset          | [[Baidu Netdisk](https://pan.baidu.com/s/1DoB68PEqjUohs8tJ7_Bv_Q), code: awtd] [[Google Drive](https://drive.google.com/file/d/1umbii5ft0Z9c6W-XBjCSs4UYBKy5k9bm/view?usp=sharing)] |

### Contact us

This repository is currently being prepared, and more details will be updated upon the acceptance of our work. If I have any questions regarding our work, please feel free to contact us at [wyb@chd.edu.cn](mailto:wyb@chd.edu.cn).