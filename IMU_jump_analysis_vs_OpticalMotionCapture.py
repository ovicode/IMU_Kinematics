
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:46:27 2023

@author: ifeoluwaolawore
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt, lfilter
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d


#________________________________________________________________________IMU processing functions_________________________________________

def Resample_accelerometer(lowg, highg):
    """
    Function to align, fuse, and extract all IMU information.
    The high-g accelerometer is interpolated to the low-g accelerometer using
    the UNIX time-stamps in the innate files. The two accelerometer are then
    aligned using a cross correlation. Typically this results in a 1 frame
    phase shift.

    Parameters:
    lowg : pandas DataFrame
        low-g data table that is extracted as raw data from Capture.U. This
        table contains both the low-g accelerometer and the gyroscope_lowgoscope.
    highg : pandas DataFrame
        low-g data table that is extracted as raw data from Capture.U.

    Returns:
    lowgtime : numpy array (Nx1)
        time (UNIX) from the low-g file
    accelerometer : numpy array (Nx3)
        X,Y,Z fused (between low-g and high-g) accelerometer from the IMU
    gyroscope_lowg : numpy array (Nx3)
        X,Y,Z gyroscope_lowgoscope from the IMU
    """
    # lowg: low-g data table (contains accelerometer and gyroscope_lowgoscope)
    # highg: high-g data table (contains only accelerometer)

    # Need to align the data
    # 1: get the data collection frequency from the low-g accelerometer
    accelerometer_lowg = lowg.iloc[:, 2:5].values
    gyroscope_lowg = lowg.iloc[:, 5:].values

    highgdata = highg.iloc[:, 2:].values

    highgtime = highg.iloc[:, 1].values
    lowgtime = lowg.iloc[:, 1].values

    index = (lowgtime < np.max(highgtime)) & (lowgtime > np.min(highgtime))
    lowgtime = lowgtime[index]
    accelerometer_lowg = accelerometer_lowg[index, :]
    gyroscope_lowg = gyroscope_lowg[index, :]

    # Create an empty array to fill with the resampled/downsampled high-g accelerometer
    resamplingHighg = np.zeros((len(lowgtime), 3))

    for jj in range(3):
        f = np.interp(lowgtime, highgtime, highgdata[:, jj])
        resamplingHighg[:, jj] = f

    # Cross-correlate the y-components
    corr_arr = np.correlate(accelerometer_lowg[:, 1], resamplingHighg[:, 1], mode='full')
    lags = np.arange(-len(accelerometer_lowg[:, 1])+1, len(resamplingHighg[:, 1]))
    lag = lags[np.argmax(corr_arr)]

    if lag > 0:
        lowgtime = lowgtime[lag+1:]
        gyroscope_lowg = gyroscope_lowg[lag+1:, :]
        accelerometer_lowg = accelerometer_lowg[lag+1:, :]
        resamplingHighg = resamplingHighg[:len(accelerometer_lowg), :]
    elif lag < 0:
        lowgtime = lowgtime[:len(lowgtime)+lag]
        gyroscope_lowg = gyroscope_lowg[:len(lowgtime), :]
        accelerometer_lowg = accelerometer_lowg[:len(lowgtime), :]
        resamplingHighg = resamplingHighg[-lag+1:, :]

    accelerometer = accelerometer_lowg

    # Find when the data is above/below 16G and replace with high-g accelerometer
    for jj in range(3):
        index = np.abs(accelerometer[:, jj]) > (9.81*16-0.1)
        accelerometer[index, jj] = resamplingHighg[index, jj]
    return lowgtime, accelerometer, gyroscope_lowg


def align_imu_signals(ref_accel, ref_gyro, ref_time, target_accel, target_gyro, target_time):
    """
    Aligns target IMU signals to reference IMU signals using cross-correlation.
    
    """
    
    # Performing cross-correlation on one component to find the lag
    corr_arr = np.correlate(ref_accel[:, 1], target_accel[:, 1], mode='full')
    lags = np.arange(-len(ref_accel[:, 1]) + 1, len(target_accel[:, 1]))
    lag = lags[np.argmax(corr_arr)]
    
    if lag > 0:
        aligned_accel = target_accel[lag+1:, :]
        aligned_gyro = target_gyro[lag+1:, :]
        aligned_time = target_time[lag+1:]
    elif lag < 0:
        len_lag = len(target_accel) + lag
        aligned_accel = target_accel[:len_lag, :]
        aligned_gyro = target_gyro[:len_lag, :]
        aligned_time = target_time[:len_lag]
    else:
        aligned_accel = target_accel
        aligned_gyro = target_gyro
        aligned_time = target_time
        
    return aligned_accel, aligned_gyro, aligned_time,lag



def detect_jumps(accel_data, takeoff_threshold=20, flight_phase_threshold=-9.81, landing_threshold=20, min_flight_time=0.1, max_flight_time=0.5, sampling_rate=1125,window_size = 500):
    """
    Detect jump events like takeoff, landing, mid-land events using vertical acceleration data. 
    Output - event indices throughtout the trial. 
    """
    max_samples_between_peaks = int(max_flight_time * sampling_rate)
    samples_to_skip = 200
    
    i = 0
    takeoff_times = []
    landing_times = []
    foot_flat_times = []
    takeoff_indices = []
    landing_indices = []
    foot_flat_indices = []
    peak_accelerations_takeoff = []
    # valid_foot_flat = []
    peak_accelerations_landing = []
    flight_durations = []

    while i < len(accel_data) - 1:
        # Check for take-off condition
        if accel_data[i] > takeoff_threshold:
            start_index = i
            
            # Find the peak between this and the next value below the flight phase threshold
            while i < len(accel_data) - 1 and accel_data[i] > flight_phase_threshold:
                i += 1
            peak_takeoff = max(accel_data[start_index:i])
            potential_landing_start = i
            
            # Check within the time window for the next peak value above the landing threshold
            end_index = i + max_samples_between_peaks
            if end_index > len(accel_data):
                end_index = len(accel_data)
            potential_landing_data = accel_data[potential_landing_start:end_index]
            
            if any(val > landing_threshold for val in potential_landing_data):
                landing_peak_index = np.argmax(potential_landing_data) + potential_landing_start
                peak_landing = accel_data[landing_peak_index]
                
                time_diff = (landing_peak_index - start_index) / sampling_rate
                if min_flight_time <= time_diff <= max_flight_time:
                    takeoff_times.append(start_index / sampling_rate)
                    landing_times.append(landing_peak_index / sampling_rate)
                    takeoff_indices.append(start_index)
                    landing_indices.append(landing_peak_index)
                    peak_accelerations_takeoff.append(peak_takeoff)
                    peak_accelerations_landing.append(peak_landing)
                    flight_durations.append(time_diff)
                    
                    
                    # Detecting foot flat after landing using the derivative method
                    # Define a window size around the landing peak
                    start_window = start_index-window_size
                    end_window = start_index
                    
                    # Extract the potential segment for foot flat
                    potential_flat_segment = accel_data[start_window:end_window]
                    
                    # Compute the derivative of the potential segment
                    derivative = np.diff(potential_flat_segment)
                    
                # Find the index where the derivative is closest to zero
                    foot_flat_index = np.argmin(np.abs(derivative)) + start_window + 1  # +1 because diff reduces the length by 1
                    foot_flat_indices.append(foot_flat_index)
                    foot_flat_times.append(foot_flat_index / sampling_rate)
                    
                    
                    i = landing_peak_index + samples_to_skip  # Skip ahead to avoid detections within 2 seconds
                    continue
            i += 1
        else:
            i += 1
    
    # Create a dataframe with the results
    jump_data = pd.DataFrame({
        'Takeoff Time (s)': takeoff_times,
        'Landing Time (s)': landing_times,
        'Foot Flat Time (s)': foot_flat_times,
        'Takeoff Index': takeoff_indices,
        'Landing Index': landing_indices,
        'Foot Flat Index': foot_flat_indices,
        'Peak Acceleration Takeoff (m/s^2)': peak_accelerations_takeoff,
        # 'Valid Foot Flat': valid_foot_flat,
        'Peak Acceleration Landing (m/s^2)': peak_accelerations_landing,
        'Flight Duration (s)': flight_durations
    })

    return jump_data


def plot_angles(angle_list, title):
    """
    Plots the given angle list.
    angle_list: List of (n, 3) arrays for representing each jump angle.
    title: Title for the plots.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))  # 3 plots: for roll, pitch, and yaw
    
    # Assuming angles are stored as [roll, pitch, yaw]
    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['r', 'g', 'b']
    
    # Iterate over each axis (Roll, Pitch, Yaw)
    for i, ax in enumerate(axes):
        for j, jump in enumerate(angle_list):  # Iterate over each jump
            ax.plot(jump[:, i], color=colors[i], label=f"Jump {j+1}" if i == 0 else "")
        
        ax.set_ylabel(labels[i])
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1].set_xlabel('Time (data points within jump)')
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()
    
def fft_50cutoff(var,landings,t):
    """
    Function to crop the intended variable into strides, pad the strides with 
    zeros and perform the FFT on the variable of interest

    Parameters
    ----------
    var : list or numpy array
        Variable of interest
    landings : list
        foot-contact or landing indices

    Returns
    -------
    freq50

    """
    # Frequency of the signal
    freq = 1/np.mean(np.diff(t))
    
    freq50 = []
    # Index through the strides
    for ii in range(len(landings)-1):
        # Zero-Pad the Variable
        intp_var = np.zeros(5000)
        intp_var[0:landings[ii+1]-landings[ii]] = var[landings[ii]:landings[ii+1]]
        fft_out = fft(intp_var)
        
        xf = fftfreq(5000,1/freq)
        # Only look at the positive
        idx = xf > 0
        fft_out = abs(fft_out[idx])
        xf = xf[idx]
        
        # Find the frequency cut-off for 50% of the signal power
        dum = cumtrapz(fft_out)
        dum = dum/dum[-1]
        idx = np.where(dum > 0.5)[0][0]
        
        freq50.append(xf[idx])
        
    return freq50

def find_lowest_frequency(gyro_segment, t_segment):
    freq = 1 / np.mean(np.diff(t_segment))
    n = len(gyro_segment)
    
    # FFT on each of the x, y, z components
    fft_x = np.abs(fft(gyro_segment[:, 0]))
    fft_y = np.abs(fft(gyro_segment[:, 1]))
    fft_z = np.abs(fft(gyro_segment[:, 2]))
    
    # Frequencies corresponding to each FFT component
    xf = fftfreq(n, 1/freq)
    
    # Consider only positive frequencies and remove zero frequency
    positive_freqs = xf[xf > 0]
    fft_x = fft_x[xf > 0]
    fft_y = fft_y[xf > 0]
    fft_z = fft_z[xf > 0]
    
    # Find the lowest non-zero frequency for each component
    min_freq_x = positive_freqs[np.argmax(fft_x)]
    min_freq_y = positive_freqs[np.argmax(fft_y)]
    min_freq_z = positive_freqs[np.argmax(fft_z)]
    
    # Return the minimum of the three
    return min(min_freq_x, min_freq_y, min_freq_z)


def findRotToLab(accel_vec):
    """
    Function to find the rotation of the foot flat acceleration vector to the 
    defined lab gravity coordinate system.

    Parameters
    ----------
    accel_vec : numpy array
        3x1 vector of the x,y,z acceleration at foot flat

    Returns
    -------
    theta : numpy array
        3x1 vector for the rotation from the foot flat acceleration to the lab
        coordinate system

    """
    iGvec = accel_vec
    iGvec = iGvec/np.linalg.norm(iGvec)
    # Define the lab coordinate system gravity vector
    lab_Gvec = np.array([0,0,1]).T
    # Compute the rotation matrix
    C = np.cross(lab_Gvec,iGvec)
    D = np.dot(lab_Gvec,iGvec)
    Z = np.array([[0,-C[2],C[1]],[C[2],0,-C[0]],[-C[1],C[0],0]])
    # Note: test the rotation matrix (R) that the norm of the colums and the rows is 1
    R = np.eye(3)+Z+(Z@Z)*(1-D)/np.linalg.norm(C)**2
    # Compute the rotation angles
    theta = np.zeros(3) # Preallocate
    theta[1] = np.arctan2(-R[2,0],np.sqrt(R[0,0]**2+R[1,0]**2))
    # Conditional statement in case theta[1] is +/- 90 deg (pi/2 rad)
    if theta[1] == np.pi/2:
        theta[2] = 0
        theta[0] = np.arctan2(R[0,1],R[1,1])
    elif theta[1] == -np.pi/2:
        theta[2] = 0
        theta[0] = -np.arctan2(R[0,1],R[1,1])
    else: 
        theta[2] = np.arctan2(R[1,0]/np.cos(theta[1]),R[0,0]/np.cos(theta[1]))
        theta[0] = np.arctan2(R[2,1]/np.cos(theta[1]),R[2,2]/np.cos(theta[1]))
    theta = theta*180/np.pi
    return theta


def filtIMUsigLowPass(sig_in, cut, t):
    # Set up a 2nd order low pass Butterworth filter
    freq = 1 / np.mean(np.diff(t))
    w = cut / (freq / 2)  # Normalize the frequency
    b, a = butter(2, w, 'low')
    # Filter the IMU signals
    sig_out = np.array([filtfilt(b, a, sig_in[:, jj]) for jj in range(3)]).T
    return sig_out

def filtIMUsigHighPass(sig_in, cut, t):
    # Set up a 2nd order low pass Butterworth filter
    freq = 1 / np.mean(np.diff(t))
    w = cut / (freq / 2)  # Normalize the frequency
    b, a = butter(4, w, 'high')
    # Filter the IMU signals
    sig_out = np.array([filtfilt(b, a, sig_in[:, jj]) for jj in range(3)]).T
    return sig_out

def filtIMUsigLowPassYaw(sig_in, cut, t):
    # Set up a 2nd order high pass Butterworth filter
    freq = 1 / np.mean(np.diff(t))
    w = cut / (freq / 2)  # Normalize the frequency
    b, a = butter(2, w, 'low')
    # Filter the IMU signal
    sig_out = lfilter(b, a, sig_in)
    return sig_out


def compute_gyro_integration_angle(gyro, accel_data, t, landing_indices, takeoff_indices, foot_flat_indices):
    """
    Integrate the gyro data to get the angles after applying a low-pass filter for each segment between foot contacts.
    
    Parameters:
    ...
    gyro : ndarray
        Gyroscope data for the entire session.
    accel_data : ndarray
        Accelerometer data for the entire session.
    dt : float
        The time difference between each data point.
    t : ndarray
        The time array corresponding to each data point for the entire session.
    landing_indices : list of int
        Indices corresponding to each foot contact.
    foot_flat_indices : list of int
        Indices corresponding to each foot flat within each segment.

    Returns:
    list of dict
        A list containing dictionaries for each segment with integrated angles for pitch, roll, and yaw.
    """
    # Number of frames to examine the gravity during midstance
    ff_frames = 20
    
    dt = np.mean(np.diff(t))
    
    results = []
    for start_idx, end_idx, ff_idx in zip(landing_indices[:-1], takeoff_indices[1:], foot_flat_indices):
        segment_data = {
            'pitch': [],
            'roll': [],
            'yaw': [],
            'thetai': None  # The initial orientation at foot flat
        }
        
        # Extracting segment data
        gyro_segment = gyro[start_idx:end_idx, :]
        accel_segment = accel_data[start_idx:end_idx, :]
        t_segment = t[start_idx:end_idx]
        
        # Filtering the gyroscope data only 
        gyrofreq50fft_results = []
        for col in range(gyro_segment.shape[1]):
            freq50fft_tmp = fft_50cutoff(gyro_segment[:, col], [0, len(gyro_segment)], t_segment)
            gyrofreq50fft_results.append(max(freq50fft_tmp))
        gyrocut_off = int(max(gyrofreq50fft_results) + 5)
        gyro_segment = filtIMUsigLowPass(gyro_segment, gyrocut_off, t_segment)
        
        # Filtering the accelerometer data only 
        accelfreq50fft_results = []
        for col in range(gyro_segment.shape[1]):
            freq50fft_tmp = fft_50cutoff(accel_segment[:, col], [0, len(accel_segment)], t_segment)
            accelfreq50fft_results.append(max(freq50fft_tmp))
        accelcut_off = int(max(accelfreq50fft_results) + 5)
        accel_segment = filtIMUsigLowPass(accel_segment, accelcut_off, t_segment)
        
        # Find midland index relative to the segment
        ML_idx = ff_idx - start_idx
        
        # Obtain the acceleration from foot contact to the foot flat
        acc_segment = accel_segment[ML_idx:ML_idx + ff_frames, :]
        
        # Compute the initial orientation using findRotToLab
        Fflat_accel = np.mean(acc_segment, axis=0) / np.mean(np.linalg.norm(acc_segment, axis=1))
        segment_data['thetai'] = findRotToLab(Fflat_accel)
        
        # Integrating gyro data to get angles for pitch, roll, and yaw
        segment_data['pitch'] = dt * cumtrapz(gyro_segment[:, 1], initial=0) - segment_data['thetai'][1]
        segment_data['roll'] = dt * cumtrapz(gyro_segment[:, 0], initial=0) - segment_data['thetai'][0]
        segment_data['yaw'] = dt * cumtrapz(gyro_segment[:, 2], initial=0) - segment_data['thetai'][2]
        
        # Ensure gyro angles are zero at the midstance frame
        segment_data['pitch'] -= segment_data['pitch'][ML_idx]
        segment_data['roll'] -= segment_data['roll'][ML_idx]
        segment_data['yaw'] -= segment_data['yaw'][ML_idx]
        
        # Linear drift correction based on MS_idx
        for angle_key in ['pitch', 'roll', 'yaw']:
            angle = segment_data[angle_key]
            angle -= angle[ML_idx]  # Remove the offset at midstance
            slope = angle[-1] / (len(angle) - ML_idx - 1)
            drift = slope * np.arange(len(angle))
            angle -= drift  # Remove the linear drift
            segment_data[angle_key] = angle
        
        results.append(segment_data)
        
    return results



def compute_accel_inclination_angle(accel,t, imu_type, ac_offsets, rotation_type):
    """
    Compute inclination angles for pitch and roll from accelerometer data.
    """
    # Filter the accelerometer signal
    
    accel = filtIMUsigLowPass(accel, 140, t)
    accel_x, accel_y, accel_z = accel[:, 0], accel[:, 1], accel[:, 2]

    # Initializing arrays for storing pitch and roll angles
    theta_ac = np.zeros(accel.shape[0])

    if rotation_type == "pitch":
        flag = 0
        for i in range(1, accel.shape[0]):
            # Compute inclination angle
            theta_ac[i] = -np.degrees(np.arctan2(accel_y[i], np.sqrt(accel_x[i]**2 + accel_z[i]**2))) + ac_offsets[1]

            # To prevent inversion of computed angle
            if (accel_z[i] < 0) and (imu_type == 'moving') and (flag == 0):
                ref_angle = theta_ac[i-1]
                if accel_x[i] < 0:
                    theta_ac[i] = abs(ref_angle - theta_ac[i]) + ref_angle
                else:
                    theta_ac[i] = ref_angle - abs(ref_angle - theta_ac[i])
                flag = 1

            elif (accel_z[i] < 0) and (imu_type == 'moving') and (flag == 1):
                if accel_x[i] < 0:
                    theta_ac[i] = abs(ref_angle - theta_ac[i]) + ref_angle
                else:
                    theta_ac[i] = ref_angle - abs(ref_angle - theta_ac[i])
            elif accel_z[i] > 0:
                flag = 0

    elif rotation_type == "roll":
        for i in range(1, accel.shape[0]):
            theta_ac[i] = -np.degrees(np.arctan2(accel_x[i], np.sqrt(accel_y[i]**2 + accel_z[i]**2))) + ac_offsets[0]

    elif rotation_type == "yaw":
        # Inclination angles can't be computed for yaw
        theta_ac = np.zeros(accel.shape[0])

    return theta_ac


def ComputeSegmentedAngle(accel_data, gyro_data, t, takeoff_idx, landing_idx, omega_drift, imu_type, ac_offsets, e_tol=3):
    """
    Computes the complementary filter angle using both gyroscopic and accelerometer data.
    Returns angles for roll, pitch, and yaw.
    """
    # Calculate the average difference in time intervals.
    dt = np.mean(np.diff(t))
    
    # Calculate the time constant and gamma for pitch using the drift value and error tolerance.
    tau_pitch = omega_drift[1] / e_tol
    gamma_pitch = 1 - tau_pitch / (tau_pitch + dt)
    
    # Calculate the time constant and gamma for roll using the drift value and error tolerance.
    tau_roll = omega_drift[0] / e_tol
    gamma_roll = 1 - tau_roll / (tau_roll + dt)
    
    # Initialize lists to store angles and times.
    all_angles = []
    
    # Obtain integrated gyro angles and the initial orientation for each segment
    integrated_gyro_angles = compute_gyro_integration_angle(gyro_data, accel_data, t, landing_idx,takeoff_idx,  foot_flat)

    for idx, (start, end) in enumerate(zip(landing_idx[:-1], takeoff_idx[1:])):  # Excluding the last landing and first takeoff
        segment_data = integrated_gyro_angles[idx]

        theta_gyro_pitch = segment_data['pitch']
        theta_gyro_roll = segment_data['roll']
        theta_gyro_yaw = segment_data['yaw']

        theta_accel_pitch = compute_accel_inclination_angle(accel_data[start : end, :], t[start : end], imu_type, ac_offsets, "pitch")
        theta_accel_roll = compute_accel_inclination_angle(accel_data[start : end, :], t[start : end], imu_type, ac_offsets, "roll")

        # Initialize arrays to store complementary filter angles for each axis.
        theta_cf_pitch = np.zeros(end - start)
        theta_cf_roll = np.zeros(end - start)
        theta_cf_yaw = np.zeros(end - start)

        for i in range(1, end - start):
            theta_cf_pitch[i] = (1 - gamma_pitch) * (theta_gyro_pitch[i] + gyro_data[start + i, 1] * dt) + gamma_pitch * theta_accel_pitch[i]
            theta_cf_roll[i] = (1 - gamma_roll) * (theta_gyro_roll[i] + gyro_data[start + i, 0] * dt) + gamma_roll * theta_accel_roll[i]
            theta_cf_yaw[i] = theta_gyro_yaw[i] + gyro_data[start + i, 2] * dt  # For yaw, just use gyro data

        # Stack the calculated angles for pitch, roll, and yaw for the segment.
        angles = np.vstack((theta_cf_pitch, theta_cf_roll, theta_cf_yaw)).T
        # Append the segment angles and times to the respective lists.
        all_angles.append(angles)

    # Return the list of segmented angles and times.
    return all_angles



#_______________________________________________________________________Mocap functions_________________________________________

def landings(force, fThresh):
    """
    Funcion to detect foot strike

    Parameters
    ----------
    force : Series
        vGRF
    fThresh : defined integer
        Force threshold

    Returns
    ------
    fly : list
        Indices from vGRF where force value was least at the beginning

    """
    land = []
    for value in range(len(force)-1): 
        if force[value] == 0 and force[value + 1] >= fThresh:
            land.append(value)
    return land
    

def flight(force, fThresh): 
    """
    Funcion to detect flight/takeoff

    Parameters
    ----------
    force : Series
        vGRF
    fThresh : defined integer
        Force threshold

    Returns
    ------
    fly : list
        Indices from vGRF where force value was least at the end 

    """
    fly = []
    if force[0] < fThresh:  # Adding this condition to detect the first takeoff if it starts below the threshold
        fly.append(0)
    for value in range(len(force)-1):
        if force[value] >= fThresh and force[value +1] == 0: 
            fly.append(value + 1)
    return fly

def trimlandings(landings, flights): 
    trimFlights = landings
    if flights[0] > landings[0] and len(flights) > len(landings):
        del(trimFlights[0])
    return trimFlights


def trimFlights(landing, fligh):
    if landing[0]>fligh[0]: 
        fligh.pop(0)
        return(fligh)
    else:
        return(fligh)


#________________________________________________________________________End of functions_________________________________________    

# shankhighg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S10-1-CMJ_TS-03391_2022-11-10-14-33-33_highg.csv")

# shanklowg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S10-1-CMJ_TS-03391_2022-11-10-14-33-33_lowg.csv")

# foothighg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S10-1-CMJ_TS-03399_2022-11-10-14-33-33_highg.csv")

# footlowg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S10-1-CMJ_TS-03399_2022-11-10-14-33-33_lowg.csv")

shankhighg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S10-1-Skater_TS-03391_2022-11-10-14-33-29_highg.csv")

shanklowg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S10-1-Skater_TS-03391_2022-11-10-14-33-29_lowg.csv")

foothighg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S10-1-Skater_TS-03399_2022-11-10-14-33-29_highg.csv")

footlowg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S10-1-Skater_TS-03399_2022-11-10-14-33-29_lowg.csv")



[flowgtime,faccelerometer,fgyroscope_lowg] = Resample_accelerometer(footlowg,foothighg)
[slowgtime_raw,saccelerometer_raw,sgyroscope_lowg_raw] = Resample_accelerometer(shanklowg,shankhighg)



saccelerometer, sgyroscope_lowg, slowgtime,lag = align_imu_signals(faccelerometer, fgyroscope_lowg, flowgtime, saccelerometer_raw, sgyroscope_lowg_raw, slowgtime_raw)


accel_data = pd.Series(faccelerometer[:,2])
accel_time = pd.Series(flowgtime)
shank_accel_data = pd.Series(saccelerometer[:,2])
shank_accel_time = pd.Series(slowgtime)

detected_jumps = detect_jumps(accel_data, takeoff_threshold=40, flight_phase_threshold=-9.81, landing_threshold=20, max_flight_time=1,window_size = 800)
takeoff_index = np.array(detected_jumps.iloc[:,3])
landing_index = np.array(detected_jumps.iloc[:,4])
foot_flat = np.array(detected_jumps.iloc[1:,5])

# Plot detectd jumps
plt.figure(figsize=(12, 6))
# plt.plot(shank_accel_data, label="Shank Acceleration Data")
plt.plot(accel_data, label="Foot Acceleration Data")
plt.scatter(detected_jumps["Takeoff Index"], detected_jumps["Peak Acceleration Takeoff (m/s^2)"], color='red', label="Detected Takeoff")
plt.scatter(detected_jumps["Landing Index"], detected_jumps["Peak Acceleration Landing (m/s^2)"], color='green', label="Detected Landing")
plt.scatter(detected_jumps["Foot Flat Index"], detected_jumps["Flight Duration (s)"], color='blue', label="Detected Midland")
plt.title("Vertical Acceleration Data with Detected Points")
plt.xlabel("Sample Index")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.grid(True)
plt.show()

    
footintegrated_angles = compute_gyro_integration_angle(fgyroscope_lowg, faccelerometer, flowgtime, landing_index,takeoff_index, foot_flat)
shankintegrated_angles = compute_gyro_integration_angle(sgyroscope_lowg, saccelerometer, slowgtime, landing_index,takeoff_index, foot_flat)
# shanksegment_angles = []
# footsegment_angles = []
rotation_types = ['roll', 'pitch', 'yaw']
ac_offsets = [0, 0, 90]

ftheta_accel_pitch = compute_accel_inclination_angle(faccelerometer, flowgtime, 'primary', ac_offsets, "pitch")
ftheta_accel_roll = compute_accel_inclination_angle(faccelerometer, flowgtime, 'primary', ac_offsets, "roll")


stheta_accel_pitch = compute_accel_inclination_angle(saccelerometer, slowgtime, 'secondary', ac_offsets, "pitch")
stheta_accel_roll = compute_accel_inclination_angle(saccelerometer, slowgtime, 'secondary', ac_offsets, "roll")

# fomega_drift = np.abs(np.mean(fgyroscope_lowg[0:500], axis=0))   #drift gyro rate
fomega_drift = [0.30,0.30]
# somega_drift = np.abs(np.mean(sgyroscope_lowg[0:500], axis=0))  
somega_drift = [0.30,0.30]   

foot_angles = ComputeSegmentedAngle(faccelerometer, fgyroscope_lowg, flowgtime, takeoff_index, landing_index, fomega_drift, "primary", ac_offsets)
shank_angles = ComputeSegmentedAngle(saccelerometer, sgyroscope_lowg, slowgtime, takeoff_index, landing_index, somega_drift, "secondary", ac_offsets)

    
# Assuming foot_angles_list and shank_angles_list are your lists containing the angle arrays for each jump
imu_angles_list = []
# Process each jump
for foot_angle, shank_angle in zip(foot_angles, shank_angles): 
    # Calculate differences for ankle angles
    diff_x = shank_angle[:, 0] - foot_angle[:, 0]  # Coronal plane (abduction-adduction)
    diff_y = shank_angle[:, 1] - foot_angle[:, 1]  # Sagittal plane (flexion-extension)
    diff_z = shank_angle[:, 2] - foot_angle[:, 2]  # Transverse plane (internal-external rotation)
    
    # Apply moving average filter on the ankle angle difference
    window = 6
    diff_x_smooth = np.convolve(diff_x, np.ones(window)/window, mode='valid')
    diff_y_smooth = np.convolve(diff_y, np.ones(window)/window, mode='valid')
    diff_z_smooth = np.convolve(diff_z, np.ones(window)/window, mode='valid')
    ankle_angles_array = np.column_stack((diff_y, diff_x, diff_z))
    
    # Store the calculated ankle angles
    imu_angles_list.append(ankle_angles_array)
# imu_angles_list.pop(0)

#________________________________________________________________________IMU ROM and peak angles_________________________________________
#imu
imu_rom_list = []
imu_peak_angles_list = []

for imu_angle in imu_angles_list:
    # Dictionary to store ROM and peak angles for this jump
    jump_metrics_imu = {
        'ROM': [],
        'Peak Positives': [],
        'Peak Negatives': []
    }
    
    for i in range(3):
        axis_data = imu_angle[:, i]
        rom = np.max(axis_data) - np.min(axis_data)
        peak_positive = np.max(axis_data)
        peak_negative = np.min(axis_data)

        # Append the computed metrics to the dictionary
        jump_metrics_imu['ROM'].append(rom)
        jump_metrics_imu['Peak Positives'].append(peak_positive)
        jump_metrics_imu['Peak Negatives'].append(peak_negative)

    # Append the metrics for this jump to the lists
    imu_rom_list.append(jump_metrics_imu['ROM'])
    imu_peak_angles_list.append({
        'Peak Positives': jump_metrics_imu['Peak Positives'],
        'Peak Negatives': jump_metrics_imu['Peak Negatives']
    })

#________________________________________________________________________Mocap data_________________________________________
    
fPath_mocap = "/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/MocapData/New_mocap/"
fileExt = r".txt"
directorylist = os.listdir(fPath_mocap)
data_mocap = [fName for fName in directorylist if fName.endswith(fileExt)]
fThresh = 30
freq = 200
    
# subName = []
# config1 = []
# movements = []
mocap_jumps_data = []

results_df = pd.DataFrame(columns=['Subject Name', 'Movement Style', 'MOCAP ROM Frontal', 'MOCAP ROM Sagittal', 'MOCAP ROM Transverse', 
                                   'MOCAP Peak Positive Frontal', 'MOCAP Peak Positive Sagittal', 'MOCAP Peak Positive Transverse', 
                                   'MOCAP Peak Negative Frontal', 'MOCAP Peak Negative Sagittal', 'MOCAP Peak Negative Transverse'])

# Additional columns for the imu data
imu_columns = ['IMU ROM Frontal', 'IMU ROM Sagittal', 'IMU ROM Transverse',
               'IMU Peak Positive Frontal', 'IMU Peak Positive Sagittal', 'IMU Peak Positive Transverse',
               'IMU Peak Negative Frontal', 'IMU Peak Negative Sagittal', 'IMU Peak Negative Transverse']

# Extend the columns of the results_df dataframe
results_df = pd.DataFrame(columns=results_df.columns.tolist() + imu_columns)


fName = "S10_Ankle_Skater_1 - PerformanceTestData_V2.txt"
subName = fName.split('_')[0]
movement = fName.split('_')[2].split(' ')[0]

dat = pd.read_csv(fPath_mocap+fName,sep='\t', skiprows = 8, header = 0)
df = pd.DataFrame(dat)
df = df.fillna(0)

if movement.lower() == 'skater':
    
    ZForce = df.FP4_GRF_Z * 1
    ZForce[ZForce<fThresh] = 0

    landing = landings(ZForce,fThresh)
    takeoffs = flight(ZForce, fThresh)
    steps = list(zip(landing,takeoffs))
    # landing[:] = [x for x in landing if x < takeoffs[-1]]
    # takeoffs[:] = [x for x in takeoffs if x > landing[0]]
    
if movement.lower() == 'cmj':
    ZForce = dat.FP2_GRF_Z * 1
    ZForce[ZForce<fThresh] = 0
    landing = landings(ZForce,fThresh)
    takeoffs = flight(ZForce, fThresh)
    # landing[:] = [x for x in landing if x < takeoffs[-1]]
    # takeoffs[:] = [x for x in takeoffs if x > landing[0]]
 
print(f"Processing file: {fName}")
print(f"Number of detected jumps in this file: {len(landing) - 1}")

sagittal_angle = df.RAnkleAngle_Sagittal.values

# Plot the ankle angle data
plt.figure(figsize=(12, 6))
plt.plot(sagittal_angle, label='Frontal Angle', color='gray')
plt.title(f"Ankle Frontal Angle with Landing and Takeoff Indices for {fName}")
plt.xlabel('Sample Index')
plt.ylabel('Angle (degrees)')

# Highlight the landing indices
plt.scatter(landing, sagittal_angle[landing], color='green', label='Detected Landing')

# Highlight the takeoff indices
plt.scatter(takeoffs, sagittal_angle[takeoffs], color='red', label='Detected Takeoff')

plt.legend()
plt.grid(True)
plt.show()

# jumps = zip(landing[:], takeoffs[1:])
jumps = zip(landing[:-1], takeoffs[1:])
for start, end in jumps:
    # Creating a 2D array (n, 3) for each jump where n is the number of samples
    # and each column represents frontal, sagittal, and transverse angles respectively.
    jump_angles = np.vstack([
        df.RAnkleAngle_Frontal.iloc[start:end].values,
        df.RAnkleAngle_Sagittal.iloc[start:end].values,
        df.RAnkleAngle_Transverse.iloc[start:end].values
    ]).T
    
    mocap_jumps_data.append(jump_angles)
    
#________________________________________________________________________Resampling and Comparing_________________________________________        
# Initialize a new list to hold the upsampled mocap data for each jump
upsampled_mocap_jumps = []

# Iterate over each pair of mocap and imu jumps
for mocap_data, imu_data in zip(mocap_jumps_data, imu_angles_list):

    # Getting the number of samples in each dataset
    num_samples_mocap = mocap_data.shape[0]
    num_samples_imu = imu_data.shape[0]

    # Creating an array representing the 'time' axis of each dataset, assuming even spacing
    time_mocap = np.linspace(0, 1, num_samples_mocap)
    time_imu = np.linspace(0, 1, num_samples_imu)

    # Initializing a new array to hold the upsampled mocap data for this jump
    upsampled_mocap_data = np.zeros((num_samples_imu, 3))

    # Interpolating each column (axis) separately
    for i in range(3):  # Iterate over the three columns (Frontal, Sagittal, Transverse)
        interp_function = interp1d(time_mocap, mocap_data[:, i], kind='linear', fill_value='extrapolate')
        upsampled_mocap_data[:, i] = interp_function(time_imu)

    # Append the upsampled mocap data for this jump to the list
    upsampled_mocap_jumps.append(upsampled_mocap_data)
        
#________________________________________________________________________MOCAP ROM and peak angles_________________________________________            
rom_list = []
peak_angles_list = []

idx_counter = 0

results_df = pd.DataFrame(columns=results_df.columns)
for mocap_data, imu_rom, imu_peak in zip(upsampled_mocap_jumps, imu_rom_list, imu_peak_angles_list):
    jump_metrics = {
        'ROM': [],
        'Peak Positives': [],
        'Peak Negatives': []
    }
    
    for i in range(3):
        axis_data = mocap_data[:, i]
        rom = np.max(axis_data) - np.min(axis_data)
        peak_positive = np.max(axis_data)
        peak_negative = np.min(axis_data)

        # Append the computed metrics to the dictionary
        jump_metrics['ROM'].append(rom)
        jump_metrics['Peak Positives'].append(peak_positive)
        jump_metrics['Peak Negatives'].append(peak_negative)
        
    new_row = {
'Subject Name': subName,
'Movement Style': movement,
'MOCAP ROM Frontal': jump_metrics['ROM'][0],
'MOCAP ROM Sagittal': jump_metrics['ROM'][1],
'MOCAP ROM Transverse': jump_metrics['ROM'][2],
'MOCAP Peak Positive Frontal': jump_metrics['Peak Positives'][0],
'MOCAP Peak Positive Sagittal': jump_metrics['Peak Positives'][1],
'MOCAP Peak Positive Transverse': jump_metrics['Peak Positives'][2],
'MOCAP Peak Negative Frontal': jump_metrics['Peak Negatives'][0],
'MOCAP Peak Negative Sagittal': jump_metrics['Peak Negatives'][1],
'MOCAP Peak Negative Transverse': jump_metrics['Peak Negatives'][2],
'IMU ROM Frontal': imu_rom[0],
'IMU ROM Sagittal': imu_rom[1],
'IMU ROM Transverse': imu_rom[2],
'IMU Peak Positive Frontal': imu_peak['Peak Positives'][0],
'IMU Peak Positive Sagittal': imu_peak['Peak Positives'][1],
'IMU Peak Positive Transverse': imu_peak['Peak Positives'][2],
'IMU Peak Negative Frontal': imu_peak['Peak Negatives'][0],
'IMU Peak Negative Sagittal': imu_peak['Peak Negatives'][1],
'IMU Peak Negative Transverse': imu_peak['Peak Negatives'][2]
}
    # results_df = results_df.append(new_row, ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame([new_row])]).reset_index(drop=True)

    # After processing all files, save the DataFrame to an Excel file
    results_df.to_excel("S10_Skater_ROM_and_Peak_Angles_Results.xlsx", index=False)
            

#________________________________________________________________________Plotting the angles side by side_________________________________________

axis_map = {'x': 0, 'y': 1, 'z': 2}
axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

# Ask the user which jump and axis to visualize
jump_idx = int(input("Enter jump number (1-indexed): ")) - 1
# Validate the user's input
if jump_idx < 0 or jump_idx >= min(len(upsampled_mocap_jumps), len(imu_angles_list)):
    print("Invalid jump number!")

else:
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    
    for i, axis in enumerate(['x', 'y', 'z']):
        # Extract the chosen jump and axis data
        mocap_data = upsampled_mocap_jumps[jump_idx][:, axis_map[axis]]
        imu_data = imu_angles_list[jump_idx][:, axis_map[axis]]
        
        # Plot
        axs[i].plot(mocap_data, label="Mocap Data")
        axs[i].plot(imu_data, label="IMU Data")
        axs[i].set_title(f"Jump {jump_idx} - {axis_labels[i]} comparison")
        axs[i].set_xlabel("Samples")
        axs[i].set_ylabel("Angles (degrees)")
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()














