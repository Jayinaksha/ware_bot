import numpy as np

# Rotates gyroscope measurements from the IMU frame 
# to the camera frame.
def alignSpatial(gyro,Ri):
    gyro=gyro.dot(Ri)

    return gyro


# Temporally aligns gyroscope and image timestamps.Gyroscope 
# measurements are upsampled. The sampling interval 'dt' is 
# the time difference between two consecutive row exposures.
def alignTemporal(gyro, tgyr, tf, te, tr, td, dt):

    # Convert from nanoseconds to seconds
    tgyr = 1e-9*(tgyr - tgyr[0])
    tf = 1e-9*(tf - tf[0])
    te = 1e-9*te

    tgyr = tgyr + td

    # The exposure of the first image starts
    t1 = tf[0]

    # The exposure of the last image ends
    t2 = tf[-1] + te[-1] + tr
    
    t = np.arange(t1,t2,dt)
    gyrx = np.interp(t,tgyr,gyro[:,0])
    gyry = np.interp(t,tgyr,gyro[:,1])
    gyrz = np.interp(t,tgyr,gyro[:,2])
    gyr_new = np.vstack((gyrx,gyry,gyrz)).T
    
    return gyr_new, t, tf, te

def remove_gravity_from_accel(gyro, R):  # R = 3x3xN rotation matrices
    accel_linear = np.zeros_like(gyro)
    for i in range(min(len(gyro), R.shape[2])):
        R_i = R[:,:,i]
        gravity_sensor = R_i @ np.array([0, 0, 9.81])
        accel_linear[i] =[i] - gravity_sensor
    return accel_linear



def computeLinearMotion(gyro, accel, t, velocity_mode=False):
    """Handle BOTH rotational AND constant velocity cases"""
    dt = np.diff(t)
    
    if velocity_mode: 
        print("CONSTANT VELOCITY ")
        
        if accel.shape[1] == 3:  
            velocity = np.cumsum(accel * dt[:, None], axis=0)
        else:  # Pure constant velocity input
            velocity = accel  # accel.npy contains velocity directly
        
        positions = np.cumsum(velocity * dt[:, None], axis=0)
        return positions, velocity
    
    else:  # Original rotational + accel
        quaternions = computeRotation(gyro, t)
        accel_linear = remove_gravity_from_accel(accel, quaternions)
        velocity = np.cumsum(accel_linear * dt[:, None], axis=0)
        positions = np.cumsum(velocity * dt[:, None], axis=0)
        return positions, velocity

# Constant velocity flag
def alignSpatial(accel_or_velocity, Ri, velocity_mode=False):
    if velocity_mode:
        return accel_or_velocity @ Ri.T  # Velocity in camera frame
    return accel_or_velocity @ Ri.T


# Computes blurfield B = (Bx,By). The horizontal and vertical
# components of the blur are returned as grayscale images Bx and By.

def computeLinearBlurfield(img, positions, K, velocity_mode=False):
    h, w = img.shape[:2]
    Bx, By = np.zeros((h, w)), np.zeros((h, w))
    
    total_disp = positions[-1] - positions[0]  # Total linear displacement
    
    for y in range(0, h, 8):  # Downsample for speed
        for x in range(0, w, 8):
            px = np.array([x, y, 1])
            ray = np.linalg.inv(K) @ px
            
            # CONSTANT VELOCITY: Uniform blur field
            blur_px_x = total_disp[0] * 500  # Focal scaling
            blur_px_y = total_disp[1] * 500
            
            Bx[y:y+8, x:x+8] = blur_px_x
            By[y:y+8, x:x+8] = blur_px_y
    
    print(f" Constant velocity blur: {blur_px_x:.1f}px x, {blur_px_y:.1f}px y")
    return Bx, By






# Computes positions of the camera during the image exposure 
# by integrating gyroscope readings.
def computeRotation(gyro, t):
    N = gyro.shape[0]
# Integrate gyroscope readings
    qts = np.zeros((N-1,4),dtype=np.float)
    q = np.array([1,0,0,0], dtype=np.float)
    dq_dt = np.zeros_like(q)
    
    for k in range(0,N-1):
        dt = t[k+1] - t[k]
        dq_dt[0] = -0.5*(q[1]*gyro[k,0]+q[2]*gyro[k,1]+q[3]*gyro[k,2])
        dq_dt[1] = 0.5*(q[0]*gyro[k,0]-q[3]*gyro[k,1]+q[2]*gyro[k,2])
        dq_dt[2] = 0.5*(q[3]*gyro[k,0]+q[0]*gyro[k,1]-q[1]*gyro[k,2])
        dq_dt[3] = -0.5*(q[2]*gyro[k,0]-q[1]*gyro[k,1]-q[0]*gyro[k,2])
        q = q + dq_dt*dt
        q = q / np.linalg.norm(q)
        qts[k,:] = q
        qts = qts.T


 # Quarternions to rotation matrices
    R = np.zeros((3,3,N-1), dtype=np.float)
    for k in range(0,N-1):
        R[0,0,k] = qts[0,k]**2 + qts[1,k]**2-qts[2,k]**2-qts[3,k]**2;
        R[0,1,k] = 2 * (qts[1,k] * qts[2,k] - qts[0,k] * qts[3,k]);
        R[0,2,k] = 2 * (qts[1,k] * qts[3,k] + qts[0,k] * qts[2,k]);
        R[1,0,k] = 2 * (qts[1,k] * qts[2,k] + qts[0,k] * qts[3,k]);
        R[1,1,k] = qts[0,k]**2 - qts[1,k]**2 + qts[2,k]**2 - qts[3,k]**2;
        R[1,2,k] = 2 * (qts[2,k] * qts[3,k] - qts[0,k] * qts[1,k]);
        R[2,0,k] = 2 * (qts[1,k] * qts[3,k] - qts[0,k] * qts[2,k]);
        R[2,1,k] = 2 * (qts[2,k] * qts[3,k] + qts[0,k] * qts[1,k]);
        R[2,2,k] = qts[0,k]**2 - qts[1,k]**2 - qts[2,k]**2 + qts[3,k]**2;
    return R