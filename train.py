#-*- coding:utf-8 -*-
# +
import torchvision.transforms as transforms
from diffusion_model.trainer_amp import GaussianDiffusion, Trainer
from diffusion_model.unet_style import create_model
from dataset import JPGPairImageGenerator
import argparse
import torch
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 数据与输出目录相对于本文件所在目录，这样本地/服务器、任何部署路径都能用
# 输入尺寸固定，开 cudnn 自动调优能明显加快卷积
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# -

def rescale(t):
    """归一化到 [-1, 1]。

    必须写成模块级函数：如果直接用 lambda，Windows 上 DataLoader 起子进程时
    transform 无法被序列化，会报 "Can't pickle <lambda>"。
    """
    return (t * 2) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfolder', type=str, default=os.path.join(BASE_DIR, 'data', 'Train', 'masks'))
    parser.add_argument('-l', '--targetfolder', type=str, default=os.path.join(BASE_DIR, 'data', 'Train', 'images'))
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=64)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200000)
    parser.add_argument('--timesteps', type=int, default=250)
    parser.add_argument('-r', '--resume_weight', type=str, default="")
    parser.add_argument('--save_every', type=int, default=10000,
                        help='每多少步存一次模型和采样图')
    args = parser.parse_args()

    inputfolder = args.inputfolder
    targetfolder = args.targetfolder
    input_size = args.input_size
    num_channels = args.num_channels
    num_res_blocks = args.num_res_blocks
    resume_weight = args.resume_weight

    # Define your transformations including rotating, scaling, and shifting
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomRotation(degrees=45),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(rescale),
    ])

    dataset = JPGPairImageGenerator(
        inputfolder,
        targetfolder,
        input_size=input_size,
        transform=transform
    )
    print("dataset size:", len(dataset))

    in_channels = 6
    out_channels = 3

    model = create_model(input_size, num_channels, num_res_blocks,
                         in_channels=in_channels, out_channels=out_channels).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        timesteps=args.timesteps,   # number of steps
        loss_type='l1'              # L1 or L2
    ).cuda()

    if len(resume_weight) > 0:
        weight = torch.load(resume_weight, map_location='cuda')
        diffusion.load_state_dict(weight['ema'])
        print("Model Loaded!")

    trainer = Trainer(
        diffusion,
        dataset,
        image_size=input_size,
        train_batch_size=args.batchsize,
        train_lr=1e-4,
        train_num_steps=args.epochs,          # total training steps
        gradient_accumulate_every=2,          # gradient accumulation steps
        ema_decay=0.995,                      # exponential moving average decay
        amp=True,
        save_and_sample_every=args.save_every,
        results_folder=os.path.join(BASE_DIR, 'base_SIM_style1.0_new'),
    )

    trainer.train()


if __name__ == '__main__':
    main()
