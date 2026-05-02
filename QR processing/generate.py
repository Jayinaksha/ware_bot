import numpy as np
import os
# Load camera and IMU calibration information
import calibration as calib
import IO    # For reading and writing data
from utils import *
from visualize import plotBlurVectors

if __name__ == '__main__':
    
    inpath=r"C:\Users\gomar\OneDrive\Desktop\Caliberation\calibration.py"
    outpath = r"C:\Users\gomar\OneDrive\Desktop\output_calib"
    print("Input folder: %s" %inpath)
    
    # Read gyroscope measurements and timestamps
   

    accel, taccel, gyro, tgyro = IO.load_imu_data(inpath)
    

# CONSTANT VELOCITY DETECTION
    velocity_mode = False
    if os.path.exists():
        print("🎯 CONSTANT VELOCITY MODE DETECTED!")
        velocity = np.load()  # m/s
        accel = np.zeros_like(velocity)  
        velocity_mode = True
    else:
        print("🔄 ACCELERATION MODE (original)")
        velocity = np.load()
        accel = velocity

    gyro_path = f"{data_dir}/image001_gyro.npy"
    gyro = np.load(gyro_path)
    tgyro = np.linspace(0, gyro.shape[0]/1000, gyro.shape[0])

    quaternions = computeRotation(gyro,tgyro)  
    accel_linear = remove_gravity_from_accel(accel, quaternions)  


    # Read image timestamps and exposure times
    tf, te = IO.readImageInfo(inpath)
    
    # Load calibration parameters from calibration.py
    scaling = calib.scaling # Downsample images?
    K = calib.K             # Camera intrinsics
    tr = calib.tr           # Camera readout time
    td = calib.td           # IMU-camera temporal offset
    Ri = calib.Ri           # IMU-to-camera rotation
    
    # If we downsample, we also need to scale intrinsics
    if scaling < 1.0:
        K = scaling*K
        K[2,2] = 1
        
    IO.createOutputFolders(outpath)
     
    ''' Temporal and spatial alignment of gyroscope and camera '''
    
    # Read the first image to get image dimensions
    img = IO.readImage(inpath, scaling, idx=0)
    height, width = img.shape[:2]
    
    dt = tr/height # Sampling interval
    accel_linear= alignSpatial(accel_linear, Ri)
    accel_linear, t, tf, te = alignTemporal(accel_linear, taccel, tf, te, tr, td, dt)