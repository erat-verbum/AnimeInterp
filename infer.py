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

  parser.add_argument("-i", "--input", type=str, default="./datasets/test_input_frames", help="Directory of input frames")

  parser.add_argument("-o", "--output", type=str, default="./datasets/test_output_frames", help="Directory of output frames")

  parser.add_argument("-f", "--factor", type=int, default=1, help="Number of frames to add between two of the original frames")

  args = parser.parse_args()

  input_frames_dir = args.input
  output_frames_dir = args.output
  frame_factor = args.factor

  files = sorted(glob.glob(input_frames_dir + '/**/*.png', recursive=True))
  del files[-1]

  if not os.path.exists(output_frames_dir):
    os.mkdir(output_frames_dir)

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

  input_frame = 1
  for f in tqdm(files):
    with torch.no_grad():
      filename_frame_1 = f
      filename_frame_2 = os.path.join(input_frames_dir, f'{input_frame+1:0>5d}.png')

      first_output_filename = os.path.join(output_frames_dir, f'{(input_frame-1)*(frame_factor+1)+1:0>5d}.png')
      last_output_filename = os.path.join(output_frames_dir, f'{input_frame*(frame_factor+1):0>5d}.png')

      shutil.copy(filename_frame_1, first_output_filename)
      shutil.copy(filename_frame_2, last_output_filename)

      frame1 = _pil_loader(filename_frame_1)
      frame2 = _pil_loader(filename_frame_2)
      transform1 = TF.Compose([TF.ToTensor()])
      frame1 = transform1(frame1).unsqueeze(0)
      frame2 = transform1(frame2).unsqueeze(0)

      for i in range(1, frame_factor + 1):
        output_frame_file_path = os.path.join(output_frames_dir, f"{(input_frame-1)*(frame_factor+1)+1+i:0>5d}.png")

        outputs = model(frame1.cuda(), frame2.cuda(), i/(frame_factor + 1))
        It_warp = outputs[0]
        to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)).save(output_frame_file_path)

      input_frame += 1

if __name__ == "__main__":
  main()