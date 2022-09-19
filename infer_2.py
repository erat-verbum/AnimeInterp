from types import FrameType
from PIL import Image
import models
import argparse
import torch
import torchvision.transforms as TF
import torch.nn as nn
import os
import shutil
import numpy as np
import cv2
import warnings
import numpy
from tqdm import tqdm
import glob
warnings.filterwarnings("ignore")

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("-d", "--dir", type=str, default="./datasets/test_output_frames", help="Directory of frames to interpolate")

  args = parser.parse_args()

  frames_dir = args.dir

  file_paths = sorted(glob.glob(frames_dir + '/**/*.png', recursive=True))

  # https://github.com/lisiyao21/AnimeInterp/blob/49b1ea2ee0d6637292adbb157f0ba6b0e8cadb0d/datas/AniTriplet.py#L34
  def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    with open(path, 'rb') as f:
      img = Image.open(f)
      resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
      cropped_img = resized_img.crop(cropArea) if cropArea != None else resized_img
      flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
      return flipped_img.convert('RGB')

  # https://github.com/lisiyao21/AnimeInterp/issues/8
  normalize1 = TF.Normalize([0., 0., 0.], [1.0, 1.0, 1.0])
  normalize2 = TF.Normalize([0, 0, 0], [1, 1, 1])
  trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])
  revmean = [-x for x in [0., 0., 0.]]
  revstd = [1.0 / x for x in [1, 1, 1]]
  revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
  revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
  revNormalize = TF.Compose([revnormalize1, revnormalize2])
  revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])
  to_img = TF.ToPILImage()

  #model = getattr(models, 'AnimeInterpNoCupy')(None).cuda()
  model = getattr(models, 'AnimeInterp')(None).cuda()
  model = nn.DataParallel(model)
  dict1 = torch.load("./checkpoints/anime_interp_full.ckpt")
  model.load_state_dict(dict1['model_state_dict'], strict=False)
  model.eval()

  def get_frame_num(path: os.PathLike) -> int:
    name = os.path.basename(path).replace(".png", "")
    return int(name)

  prev_frame_num = 1
  prev_file_path = file_paths[0]
  for file_path in tqdm(file_paths):
    with torch.no_grad():

      frame_num = get_frame_num(file_path)

      num_intermediary_frames = frame_num - prev_frame_num -1

      if num_intermediary_frames > 0:
        frame1 = _pil_loader(prev_file_path)
        frame2 = _pil_loader(file_path)
        transform1 = TF.Compose([TF.ToTensor()])
        frame1 = transform1(frame1).unsqueeze(0)
        frame2 = transform1(frame2).unsqueeze(0)

        for i in range(num_intermediary_frames):
          intermediary_frame_num = prev_frame_num + i + 1
          intermediary_frame_file_path = os.path.join(frames_dir, f"{intermediary_frame_num:0>10d}.png")

          outputs = model(frame1.cuda(), frame2.cuda(), i/(num_intermediary_frames + 1))
          It_warp = outputs[0]
          to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)).save(intermediary_frame_file_path)

    prev_frame_num = frame_num
    prev_file_path = file_path

if __name__ == "__main__":
  main()