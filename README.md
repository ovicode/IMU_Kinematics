# IMU_OpticalCapture_Validation
This script calculates ankle joint kinematics from dynamic jumps using IMUs compared to kinematics from conventional Optical Motion kinematics. It focuses on processing IMU (Inertial Measurement Unit) data, particularly gyroscope and accelerometer data, for biomechanical analysis.

**Resample_accelerometer**: This function aligns and fuses data from low-g and high-g accelerometers using UNIX time-stamps. It interpolates high-g data to low-g time, cross-correlates them to find alignment, and replaces low-g data with high-g data when acceleration exceeds a threshold.

**align_imu_signals**: Aligns target IMU signals to reference IMU signals by determining the time lag through cross-correlation. It adjusts the target accelerometer and gyroscope data based on the calculated lag.

**detect_jumps**: Identifies jump events (takeoff, landing, and foot-flat) from vertical acceleration data. It detects takeoff and landing based on acceleration thresholds and calculates flight durations, outputting a dataframe with event times and peak accelerations.

**plot_angles**: Plots the roll, pitch, and yaw angles of multiple jumps. It visualizes the angles in three subplots, enhancing analysis through color-coded representations for clarity.

**fft_50cutoff**: Performs a Fast Fourier Transform (FFT) on a variable of interest, cropping it into strides based on landing indices. It identifies the frequency cut-off for 50% of the signal power.

**find_lowest_frequency**: Calculates the lowest non-zero frequency of gyroscope data components (x, y, z) using FFT. It returns the minimum frequency across the three components.

**findRotToLab**: Determines the rotation of the foot flat acceleration vector relative to the lab coordinate system. It computes the rotation matrix and angles needed to align the acceleration vector with the defined gravity vector of the lab.

**Filtering Functions:**

**filtIMUsigLowPass:** Applies a low-pass Butterworth filter to the input IMU signal.
**filtIMUsigHighPass:** Applies a high-pass Butterworth filter to the input IMU signal.
**filtIMUsigLowPassYaw:** Applies a low-pass Butterworth filter specifically for yaw data.
Angle Computation:

**compute_gyro_integration_angle:** Integrates the gyroscope data to calculate pitch, roll, and yaw angles between foot contacts, adjusting for drift and correcting offsets.
compute_accel_inclination_angle: Computes inclination angles (pitch and roll) from accelerometer data.
**ComputeSegmentedAngle**: Combines gyroscope and accelerometer data using a complementary filter to compute final angles (roll, pitch, and yaw) for different segments of the data.

**Mocap Functions:**

**landings, flight**: Detect foot strike and takeoff events based on vertical ground reaction force (vGRF) data.
**trimlandings, trimFlights:** Trim the detected landing and flight indices to ensure consistency in start and end points.

**IMU Data Handling:**

Data from IMUs placed on the shank and foot is loaded, resampled, and aligned using functions like Resample_accelerometer and align_imu_signals.
The script then uses the aligned data to detect jumps, focusing on takeoff and landing phases.

**Plotting:**

The script begins to plot the detected jumps with corresponding takeoff and landing events but is incomplete.
