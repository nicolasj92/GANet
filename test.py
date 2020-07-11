import re
import sys
import numpy as np
from struct import unpack
import matplotlib.pyplot as plt

# Very simple script to visually compare .pfm files of predictions and GTs

def readPFM(file): 
    with open(file, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
    return img, height, width


if __name__ == "__main__":
    img1, height, width = readPFM("/media/nicolas/projects/rvc_devkit/stereo/datasets_middlebury2014/test/Kitti2015_000000_10/disp0GANet_RVC.pfm")
    # img2, height, width = readPFM("/media/nicolas/projects/rvc_devkit/stereo/datasets_middlebury2014/test/Kitti2015_000000_10/disp0GT.pfm")
    # img1, height, width = readPFM("/media/nicolas/projects/rvc_devkit/stereo/datasets_middlebury2014/training/ETH3D2017_delivery_area_1l/disp0GANet_RVC.pfm")
    # img2, height, width = readPFM("/media/nicolas/projects/rvc_devkit/stereo/datasets_middlebury2014/training/ETH3D2017_delivery_area_1l/disp0GT.pfm")
    # img1, height, width = readPFM("/media/nicolas/projects/rvc_devkit/stereo/datasets_middlebury2014/training/Middlebury2014_Playroom/disp0GANet_RVC.pfm")
    # img2, height, width = readPFM("/media/nicolas/projects/rvc_devkit/stereo/datasets_middlebury2014/training/Middlebury2014_Playroom/disp0GT.pfm")   
    # plt.imshow(img1-img2)
    plt.imshow(img1)
    plt.show()