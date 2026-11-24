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

## References

References cited in the paper :

1. Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., Süsstrunk, S.: Slic superpixels compared to state-of-the-art superpixel methods. IEEE transactions on pattern analysis and machine intelligence 34(11), 2274–2282 (2012)
2. Bernal, J., Sánchez, F.J., Fernández-Esparrach, G., Gil, D., Rodríguez, C., Villarino, F.: Wm-dova maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. Computerized medical imaging and graphics 43, 99–111 (2015)
3. Bui, N.T., Hoang, D.H., Nguyen, Q.T., Tran, M.T., Le, N.: Meganet: Multi-scale edge-guided attention network for weak boundary polyp segmentation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 7985–7994 (2024)
4. Chaitanya, K., Karani, N., Baumgartner, C.F., Erdil, E., Becker, A., Donati, O., Konukoglu, E.: Semi-supervised task-driven data augmentation for medical image segmentation. Medical Image Analysis 68, 101934 (2021)
5. Dong, B., Wang, W., Fan, D.P., Li, J., Fu, H., Shao, L.: Polyp-pvt: Polyp segmentation with pyramid vision transformers. CAAI Artificial Intelligence Research 2 (2023)
6. Dorjsembe, Z., Pao, H.K., Xiao, F.: Polyp-ddpm: Diffusion-based semantic polyp synthesis for enhanced segmentation. In: 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). pp. 1–7. IEEE (2024)
7. Du, Y., Jiang, Y., Tan, S., Wu, X., Dou, Q., Li, Z., Li, G., Wan, X.: Arsdm: colonoscopy images synthesis with adaptive refinement semantic diffusion models. In: International conference on medical image computing and computer-assisted intervention. pp. 339–349. Springer (2023)
8. Fan, D.P., Ji, G.P., Zhou, T., Chen, G., Fu, H., Shen, J., Shao, L.: Pranet: Parallel reverse attention network for polyp segmentation. In: International conference on medical image computing and computer-assisted intervention. pp. 263–273. Springer (2020)
9. Guo, X., Chen, Z., Liu, J., Yuan, Y.: Non-equivalent images and pixels: Confidence-aware resampling with meta-learning mixup for polyp segmentation. Medical image analysis 78, 102394 (2022)
10. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)
11. Jha, D., Smedsrud, P.H., Riegler, M.A., Halvorsen, P., De Lange, T., Johansen, D., Johansen, H.D.: Kvasir-seg: A segmented polyp dataset. In: MultiMedia modeling: 26th international conference, MMM 2020, Daejeon, South Korea, January 5–8, 2020, proceedings, part II 26. pp. 451–462. Springer (2020)
12. Lin, D., Cao, Y., Zhu, W., Li, Y.: Few-shot defect segmentation leveraging abundant defect-free training samples through normal background regularization and crop-and-paste operation. In: 2021 IEEE International Conference on Multimedia and Expo (ICME). pp. 1–6. IEEE Computer Society (2021)
13. Lou, A., Guan, S., Loew, M.: Caranet: context axial reverse attention network for segmentation of small medical objects. Journal of Medical Imaging 10(1), 014005–014005 (2023)
14. Ma, Y., Liu, Y., Cheng, J., Zheng, Y., Ghahremani, M., Chen, H., Liu, J., Zhao, Y.: Cycle structure and illumination constrained gan for medical image enhancement. In: International conference on medical image computing and computer-assisted intervention. pp. 667–677. Springer (2020)
15. Qiu, K., Gao, Z., Zhou, Z., Sun, M., Guo, Y.: Noise-consistent siamese-diffusion for medical image synthesis and segmentation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15672–15681 (2025)
16. Qiu, R., Xia, K., Gao, F., Yang, S., Cai, D., Wang, J., Chen, Y., Wang, L.: Accurate boundary alignment and realism enhancement for colonoscopic polyp image-mask pair generation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 34–44. Springer (2025)
17. Sandfort, V., Yan, K., Pickhardt, P.J., Summers, R.M.: Data augmentation using generative adversarial networks (cyclegan) to improve generalizability in ct segmentation tasks. Scientific reports 9(1), 16884 (2019)
18. Sharma, V., Kumar, A., Jha, D., Bhuyan, M.K., Das, P.K., Bagci, U.: Controlpolypnet: Towards controlled colon polyp synthesis for improved polyp segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2325–2334 (2024)
19. Silva, J., Histace, A., Romain, O., Dray, X., Granado, B.: Toward embedded detection of polyps in wce images for early diagnosis of colorectal cancer. International journal of computer assisted radiology and surgery 9, 283–293 (2014)
20. Ta, N., Chen, H., Lyu, Y., Wu, T.: Ble-net: boundary learning and enhancement network for polyp segmentation. Multimedia Systems 29(5), 3041–3054 (2023)
21. Tajbakhsh, N., Gurudu, S.R., Liang, J.: Automated polyp detection in colonoscopy videos using shape and context information. IEEE transactions on medical imaging 35(2), 630–644 (2015)
22. Thambawita, V., Salehi, P., Sheshkal, S.A., Hicks, S.A., Hammer, H.L., Parasa, S., Lange, T.d., Halvorsen, P., Riegler, M.A.: Singan-seg: Synthetic training data generation for medical image segmentation. PloS one 17(5), e0267976 (2022)
23. Vázquez, D., Bernal, J., Sánchez, F.J., Fernández-Esparrach, G., López, A.M., Romero, A., Drozdzal, M., Courville, A.: A benchmark for endoluminal scene segmentation of colonoscopy images. Journal of healthcare engineering 2017(1), 4037190 (2017)
24. Wei, J., Hu, Y., Zhang, R., Li, Z., Zhou, S.K., Cui, S.: Shallow attention network for polyp segmentation. In: International conference on medical image computing and computer-assisted intervention. pp. 699–708. Springer (2021)
25. Yin, Z., Liang, K., Ma, Z., Guo, J.: Duplex contextual relation network for polyp segmentation. In: 2022 IEEE 19th international symposium on biomedical imaging (ISBI). pp. 1–5. IEEE (2022)
26. Yu, J., Qi, L.: Safe-net: Shape-aware and feature enhancement network for polyp segmentation. Biomedical Signal Processing and Control 99, 106906 (2025)
27. Zhang, R., Li, G., Li, Z., Cui, S., Qian, D., Yu, Y.: Adaptive context selection for polyp segmentation. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part VI 23. pp. 253–262. Springer (2020)
28. Zhao, X., Zhang, L., Lu, H.: Automatic polyp segmentation via multi-scale subtraction network. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part I 24. pp. 120–130. Springer (2021)
29. Zhao, Y., Zhou, T., Gu, Y., Zhou, Y., Zhang, Y., Wu, Y., Fu, H.: Weakpolyp-sam: Segment anything model-driven weakly-supervised polyp segmentation. Knowledge-Based Systems p. 113701 (2025)
30. Zhou, T., Zhang, Y., Chen, G., Zhou, Y., Wu, Y., Fan, D.P.: Edge-aware feature aggregation network for polyp segmentation. Machine Intelligence Research 22(1), 101–116 (2025)

## Acknowledgement

The diffusion UNet is built on [guided-diffusion](https://github.com/openai/guided-diffusion), and the data partitioning follows the [PraNet](https://github.com/DengPingFan/PraNet) benchmark, thanks to the authors.
