
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import utils
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Lambda

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from diffusion_model.trainer import GaussianDiffusion       # noqa: E402
from diffusion_model.unet_style import create_model         # noqa: E402

INPUT_SIZE = 256
NUM_CHANNELS = 64
NUM_RES_BLOCKS = 2
TIMESTEPS = 250


def build_transform():
    # 与训练/官方 inference.py 完全一致：转 PIL -> resize 256 -> tensor -> 映射到 [-1,1]
    return Compose([ToPILImage(), Resize((INPUT_SIZE, INPUT_SIZE)), ToTensor(),
                    Lambda(lambda t: (t * 2) - 1)])


def load_mask(path, tf):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return tf(np.array(img)).unsqueeze(0)      # [1,3,256,256]


def centroid(path, grid=64):
    """返回息肉质心的相对坐标 (x,y)，用于判断位置是否分散"""
    m = np.array(Image.open(path).convert('L').resize((grid, grid)))
    ys, xs = np.nonzero(m > 127)
    if len(xs) == 0:
        return None
    return np.array([xs.mean() / grid, ys.mean() / grid])


def pick_spread(mask_dir, n, seed=0):
    """最远点采样：每次挑一个离已选集合最远的息肉位置"""
    files = sorted(os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    cand = [(f, centroid(f)) for f in files]
    cand = [(f, c) for f, c in cand if c is not None]
    rng = np.random.RandomState(seed)
    chosen = [cand.pop(rng.randint(len(cand)))]
    while len(chosen) < n and cand:
        best, best_d = None, -1.0
        for f, c in cand:
            d = min(float(np.linalg.norm(c - cc)) for _, cc in chosen)
            if d > best_d:
                best_d, best = d, (f, c)
        chosen.append(best)
        cand.remove(best)
    return [f for f, _ in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask-dir', default=None, help='mask 目录；不给则自动从 data/masks 挑位置分散的')
    ap.add_argument('--weight', default=r'D:\base_SIM_style1.0_new\model-20.pt')
    ap.add_argument('--out', default=os.path.join(BASE_DIR, '_samples_out'))
    ap.add_argument('--n', type=int, default=20, help='生成多少张（mask 数量）')
    ap.add_argument('--num-samples', type=int, default=1, help='每张 mask 生成几张')
    ap.add_argument('--batchsize', type=int, default=4, help='批大小，显存不够就调小')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    img_out = os.path.join(args.out, 'images')
    msk_out = os.path.join(args.out, 'masks')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    # ---------- 1) 选 mask ----------
    if args.mask_dir:
        mask_paths = sorted(os.path.join(args.mask_dir, f) for f in os.listdir(args.mask_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')))[:args.n]
        print('使用指定 mask 目录：%s（%d 张）' % (args.mask_dir, len(mask_paths)))
    else:
        mask_paths = pick_spread(os.path.join(BASE_DIR, 'data', 'Train', 'masks'), args.n, args.seed)
        print('自动挑选 %d 张"息肉位置分散"的 mask：' % len(mask_paths))
        for p in mask_paths:
            c = centroid(p)
            print('   %-14s 息肉质心 x=%.2f y=%.2f' % (os.path.basename(p), c[0], c[1]))

    # ---------- 2) 建模型 + 载入 EMA 权重 ----------
    print('\n加载模型与权重 ...')
    model = create_model(INPUT_SIZE, NUM_CHANNELS, NUM_RES_BLOCKS,
                         in_channels=6, out_channels=3).cuda()
    diffusion = GaussianDiffusion(model, image_size=INPUT_SIZE,
                                  timesteps=TIMESTEPS, loss_type='l1').cuda()
    ckpt = torch.load(args.weight, map_location='cuda')
    diffusion.load_state_dict(ckpt['ema'])
    diffusion.eval()
    print('已加载：%s（训练到第 %s 步的 EMA 权重）' % (args.weight, ckpt.get('step')))

    # ---------- 3) 分批采样 ----------
    tf = build_transform()
    all_imgs, t0, cnt = [], time.time(), 0
    total = len(mask_paths) * args.num_samples
    for start in range(0, len(mask_paths), args.batchsize):
        group = mask_paths[start:start + args.batchsize]
        cond = torch.cat([load_mask(p, tf) for p in group], 0).cuda()
        for k in range(args.num_samples):
            with torch.no_grad():
                imgs = diffusion.sample(batch_size=len(group), condition_tensors=cond)
            imgs = (imgs + 1) * 0.5
            for j, p in enumerate(group):
                stem = os.path.splitext(os.path.basename(p))[0]
                name = '%02d_%s' % (start + j + 1, stem)
                if args.num_samples > 1:
                    name += '_s%d' % (k + 1)
                utils.save_image(imgs[j], os.path.join(img_out, name + '.png'))
                utils.save_image(cond[j], os.path.join(msk_out, name + '.png'))
                all_imgs.append(imgs[j].cpu())
                cnt += 1
        el = time.time() - t0
        print('  已生成 %d/%d（耗时 %.0f 秒，平均 %.1f 秒/张）' % (cnt, total, el, el / max(cnt, 1)))

    # ---------- 4) 总览拼图 ----------
    utils.save_image(torch.stack(all_imgs), os.path.join(args.out, 'grid_images.png'), nrow=4)
    utils.save_image(torch.stack([load_mask(p, tf)[0] for p in mask_paths]),
                     os.path.join(args.out, 'grid_masks.png'), nrow=4)
    print('\n完成！共 %d 张' % cnt)
    print('  生成图   : %s' % img_out)
    print('  条件 mask: %s' % msk_out)
    print('  总览拼图 : %s' % os.path.join(args.out, 'grid_images.png'))


if __name__ == '__main__':
    main()