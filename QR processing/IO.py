import argparse
import numpy as np
import os
import glob
from PIL import Image

def parseInputs():
    desc = "Estimate gyro-based blur fields."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', type=str, metavar='', required=True, help='input folder')
    parser.add_argument('-o', '--output', type=str, metavar='', required=True, help='output folder')
    args = parser.parse_args()
    return args.input, args.output


def load_imu_data(imu_path):
    
    data = np.loadtxt(imu_path)
    
    # Accelerometer data (type 1)
    accel_mask = data[:,0] == 1
    accel = data[accel_mask, 2:5]  # [ax, ay, az]
    taccel = data[accel_mask, 1] * 1e-9
    
    # Gyroscope data (type 4) - for orientation purpose
    gyro_mask = data[:,0] == 4
    gyro = data[gyro_mask, 2:5]  # [wx, wy, wz]
    tgyro = data[gyro_mask, 1] * 1e-9
    
    return accel, taccel, gyro, tgyro
    
def readImageInfo(inpath):
    datapath = inpath + '/images/images.json'
    info = np.loadtxt(datapath, dtype='float_', delimiter=' ')
    
    # Extract timestamps and exposure times
    tf = info[:,0]
    te = info[:,1]
    
    return tf, te

def readImage(inpath, scaling, idx):

    fnames = []
    for ext in ('*.jpg', '*.png'):
        datapath = inpath + '/images/' + ext
        fnames.extend(sorted(glob.glob(datapath)))
    
    if idx < len(fnames):
        img = Image.open(fnames[idx])
    else:
        raise ValueError('Could not read image with index: %d' %idx)
    
    # Downsample
    if scaling < 1.0:
        w = int(scaling*img.size[0])
        h = int(scaling*img.size[1])
        img = img.resize((w,h), resample=Image.BICUBIC)
        
    img = np.array(img)
    
    return img
    
    
def writeImage(img, outpath, folder, idx):
    
    fname = '%04d.png' %(idx)
    img = Image.fromarray(img.astype(np.uint8))
    img.save(outpath + '/' + folder + '/' + fname)
    
def createOutputFolders(outpath):

    try:
        os.makedirs(outpath + '/blurred')
        os.makedirs(outpath + '/blurx')
        os.makedirs(outpath + '/blury')
        os.makedirs(outpath + '/visualization/')
    except FileExistsError:
        # Directory already exists
        pass