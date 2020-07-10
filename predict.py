from __future__ import print_function
import argparse
import scipy
import skimage
import skimage.io
import skimage.transform
import imageio
import struct
import time
from PIL import Image
from math import log10
#from GCNet.modules.GCNet import L1Loss
import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from models.GANet_deep import GANet
from dataloader.data import get_test_set
import numpy as np

import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--data_path', type=str, required=True, help="data root")
parser.add_argument('--test_list', type=str, required=True, help="training list")
parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")
parser.add_argument('--name', type=str, default='GANet_RVC', help='algorithm name for submission')

opt = parser.parse_args()


print(opt)
if opt.model == 'GANet11':
    from models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from models.GANet_deep import GANet
else:
    raise Exception("No suitable model found ...")
    
cuda = opt.cuda
#cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#torch.manual_seed(opt.seed)
#if cuda:
#    torch.cuda.manual_seed(opt.seed)
#print('===> Loading datasets')


print('===> Building model')
model = GANet(opt.max_disp)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
       
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


# Converts a string to bytes (for writing the string into a file). Provided for
# compatibility with Python 2 and 3.
def StrToBytes(text):
    if sys.version_info[0] == 2:
        return text
    else:
        return bytes(text, 'UTF-8')

# Writes a .pfm file containing a disparity image according to Middlebury format.
# Expects pixels as a list of floats
def WriteMiddlebury2014PfmFile(path, width, height, disp_data):
    print(disp_data.shape)
    disp_data = np.flip(disp_data, axis=1)
    disp_float = disp_data.flatten().tolist()[::-1]
    path = path.replace("png", "pfm")

    # print(width, height)
    # import matplotlib.pyplot as plt
    # plt.imshow(disp_data)
    # plt.show()

    with open(path, 'wb') as pfm_file:
        pfm_file.write(StrToBytes('Pf\n'))
        pfm_file.write(StrToBytes(str(width) + ' ' + str(height) + '\n'))
        pfm_file.write(StrToBytes('-1\n'))  # negative number means little endian
        pfm_file.write(struct.pack('<' + str(len(disp_float)) + 'f', *disp_float))  # < means using little endian

def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h != crop_height or w != crop_width:
        zoom_h = crop_height / h
        zoom_w = crop_width / w
        
        temp_data = scipy.ndimage.zoom(temp_data, (1, zoom_h, zoom_w), order=1)

        h = crop_height
        w = crop_width

    # if h <= crop_height and w <= crop_width:
    #     temp = temp_data
    #     temp_data = np.zeros([6, crop_height, crop_width], 'float32')
    #     temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    # else:
    #     start_x = int((w - crop_width) / 2)
    #     start_y = int((h - crop_height) / 2)
    #     temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def test_post_processing(pred, target_height, target_width):
    zoom_h = target_height / pred.shape[0]
    zoom_w = target_width / pred.shape[1]
        
    pred = scipy.ndimage.zoom(pred, (zoom_h, zoom_w), order=1)
    pred = pred * zoom_w
    return pred


def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]	
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data, size

def test(leftname, rightname, savename_disp, savename_runtime):
  #  count=0
    data, size = load_data(leftname, rightname)
    input1, input2, height, width = test_transform(data, opt.crop_height, opt.crop_width)

    
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        start = time.time()
        prediction = model(input1, input2)
        end = time.time()
        execution_time = np.round(end-start, 2)
     
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    
    # rescale to target resolution
    temp = test_post_processing(temp, size[0], size[1])

    # write image
    WriteMiddlebury2014PfmFile(savename_disp, size[1], size[0], temp)
    with open(savename_runtime, 'w') as f:
        f.writelines(str(execution_time))


   
if __name__ == "__main__":
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    for index in range(len(filelist)):
        current_file = filelist[index]
        leftname = os.path.join(file_path, current_file[0: len(current_file) - 1], "im0.png")
        rightname = os.path.join(file_path, current_file[0: len(current_file) - 1], "im1.png")
        print("Processing file: {}".format(leftname))
        
        savename_disp = os.path.join(file_path, current_file[0: len(current_file) - 1], "disp0{}.pfm".format(opt.name))
        savename_runtime = os.path.join(file_path, current_file[0: len(current_file) - 1], "time{}.txt".format(opt.name))
        
        print("Saving result to file: {}".format(savename_disp))
        
        test(leftname, rightname, savename_disp, savename_runtime)
        

