import numpy as np

# Downsampling factor. For example, the image
# resolution is halved when scaling is set to 0.5
scaling = 0.5

# Camera intrinsics [fx 0 cx; 0 fy cy; 0 0 1] at
# the original resolution.
K = np.array([[1558.6899, 0, 939.6533],
			  [0, 1558.6899, 518.4131],
			  [0, 0, 1]])
			  
# Rotation from IMU frame to camera frame. 
# Obtained using the Android documentation.
Ri = np.array([[0, 1, 0],
			   [1, 0, 0],
			   [0, 0, 1]])

# Camera readout time (rolling shutter skew) in seconds. The 
# time difference between the first and last row exposure. Can 
# be obtained using the Android camera2 API or via calibration.
tr = 0.0244944

# IMU-camera temporal offset in seconds. If 'td' is set to zero,
# it is assumed that the first gyroscope measurement in 'imu.txt' 
# corresponds to the start of the first image exposure in 'images.txt'.
td = 0.022

# In calibration.py, update input expectations:
def calibrate_imu(imu_path):
    data = np.loadtxt(imu_path)
    accel_data = data[data[:,0] == 1, 2:5]  # Type 1
    gyro_data = data[data[:,0] == 4, 2:5]   # Type 4
    
    accel_bias = np.mean(accel_data, axis=0)
    gyro_bias = np.mean(gyro_data, axis=0)
    
    return {'accel_bias': accel_bias, 'gyro_bias': gyro_bias}
