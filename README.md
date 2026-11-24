# SFD-Polyp

Paper has been accepted by MICCAI 2026 and is also available at [Paper](https://papers.miccai.org/miccai-2026/0557-Paper4818.html) and [PDF](https://papers.miccai.org/miccai-2026/paper/4818_paper.pdf).

Clone this repository

```
git clone https://github.com/zhaiying0925/SFD-polyp.git
cd SFD-polyp
```

## Environments Preparation

My python version: 3.8.10, cuda version: 12.1

```
conda create -n SFD-polyp python=3.8
conda activate SFD-polyp
```

```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

```
pip install -r requirements.txt
```

## Dataset Preparation

| Dataset           | Download link                                                                          | Source                                                          | Usage       |
|:-----------------:|:--------------------------------------------------------------------------------------:|:---------------------------------------------------------------:|:-----------:|
| Kvasir-SEG        | [Google Drive](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view) | [Link](https://datasets.simula.no/kvasir-seg/)                  | Train (900) |
| CVC-ClinicDB      | [Google Drive](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view) | [Link](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/) | Train (550) |
| CVC-ColonDB       | [Google Drive](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view) | [Link](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/) | Test (380)  |
| CVC-300           | [Google Drive](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view) | [Link](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/) | Test (60)   |
| ETIS-LaribPolypDB | [Google Drive](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view) | [Link](https://polyp.grand-challenge.org/ETISLarib/)            | Test (196)  |
| Kvasir-SEG        | [Google Drive](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view) | [Link](https://datasets.simula.no/kvasir-seg/)                  | Test (100)  |
| CVC-ClinicDB      | [Google Drive](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view) | [Link](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/) | Test (62)   |

Note that to be consistent with the experimental setup, we follow the data partitioning scheme of the PraNet benchmark: 900 images from Kvasir-SEG and 550 images from CVC-ClinicDB are used as the training set (```data/Train```), while the remaining samples along with the other three complete datasets (CVC-ColonDB, CVC-300, and ETIS-LaribPolypDB) serve as the test set (```data/test```, only required for evaluating downstream polyp segmentation models).  
The Google Drive links are the ready-split ```TrainDataset``` and ```TestDataset``` bundles provided by the PraNet benchmark; the ```TestDataset``` bundle (327MB) contains all three test datasets as well as the held-out test images of Kvasir-SEG and CVC-ClinicDB, and the ```TrainDataset``` bundle (400MB) contains the 1450 training pairs used in this repository.  
The directory for the datasets should be as following:  

```
📁SFD-polyp/
└─── 📁data/
     ├─── 📁Train/                          # training set: 1450 paired images and masks
     │    ├─── 📁images/
     │    │    ├─── 📄1.png ~ 550.png       # CVC-ClinicDB (550)
     │    │    ├─── 📄cju0qx73cjw57...png   # Kvasir-SEG (900, original filenames)
     │    │    └─── 📄...
     │    └─── 📁masks/                     # same filenames as images/
     │         ├─── 📄1.png
     │         └─── 📄...
     └─── 📁test/                           # test set: 798 images from 5 datasets
          ├─── 📁Kvasir/                    # 100, each contains 📁images/ + 📁masks/
          ├─── 📁CVC-ClinicDB/              # 62
          ├─── 📁CVC-ColonDB/               # 380
          ├─── 📁CVC-300/                   # 60
          └─── 📁ETIS-LaribPolypDB/         # 196
```

## Train from scratch

Before training and testing, please make sure the dataset paths are correct. You can change them via `-i` (mask folder) and `-l` (image folder) in `train.py`, or just keep the default `data/Train/masks` and `data/Train/images`.

```python
# Step 1: generate diversified condition masks (SLIC superpixel + BFS growth)
python process.py
```

```python
# Step 2: train the style-fused diffusion model from scratch (250-step DDPM, L1 loss, 200k steps, checkpoint saved every 10000 steps)
python train.py --save_every 10000
```

## Evaluate

Before generating synthetic polyp images, please download the pretrained model by the following link and put it in the ```base_SIM_style1.0_new``` folder.  

| Model                             | Download Link                                                         |
|:---------------------------------:|:---------------------------------------------------------------------:|
| Diffusion model (200k steps, EMA) | [Pan Baidu](https://pan.baidu.com/s/1FpWC3AoD8JvkeuaerBqdgw?pwd=6677) |

Generate synthetic polyp images from condition masks (250-step reverse diffusion, 256×256 output):

```python
python inference.py -w base_SIM_style1.0_new/model-20.pt -i processed/masks -e generated
```

## Results

We retrained the Polyp-PVT segmentation model on a **1:1 mixture** of real and synthetic polyp images (1450 real + 1450 generated by SFD-Polyp, with the synthetic masks as ground truth), and evaluated it on the five test sets in ```data/test``` against the officially released Polyp-PVT weights (trained on the 1450 real images only). Each cell reports Dice/IoU in %.

Note that the training and validation data of the segmentation model are exactly the same datasets described above: the 1450 real training images are the pairs in ```data/Train```, and validation is performed on the five test sets in ```data/test``` after each training epoch.

| Method                             | Kvasir        | CVC-ClinicDB  | CVC-ColonDB | CVC-300   | ETIS-LaribPolypDB |
|:----------------------------------:|:-------------:|:-------------:|:-----------:|:---------:|:-----------------:|
| Polyp-PVT (real only)              | 91.9/86.6     | 94.0/89.3     | 80.6/72.6   | 90.2/83.4 | 79.4/71.3         |
| + SFD-Polyp (1:1 real : synthetic) | **98.9/97.8** | **98.1/96.5** | 79.2/71.9   | 89.5/82.8 | 78.5/70.7         |

The retrained segmentation model can be downloaded here: [Pan Baidu](https://pan.baidu.com/s/1WhCKL3a9kdZqR3eVDMbfJA?pwd=2233)

## Citation

If there is any help for your research, please star this repository and if you want to follow this work, you can cite as follows:

```md
@InProceedings{HanLon_Learning_MICCAI2026,
  author = { Han, Longfei AND Zhai, Ying AND Li, Haisheng},
  title = { { Learning Diverse and Realistic Polyp Data via Style-Fused Diffusion Models } },
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year = {2026},
  publisher = {Springer Nature Switzerland},
  volume = {LNCS 16892},
  month = {September},
  doi = {10.1007/978-3-032-38233-7_25},
  page = {pending}
}
```

## Acknowledgement

The diffusion UNet is built on [guided-diffusion](https://github.com/openai/guided-diffusion), and the data partitioning follows the [PraNet](https://github.com/DengPingFan/PraNet) benchmark, thanks to the authors.
