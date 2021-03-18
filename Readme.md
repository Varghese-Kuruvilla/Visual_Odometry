# File Utils for Creating Sequences for Parking Spot

This branch provides helper codes to create different sequences, each sequence containing synchronized color and depth image streams.
The different folders:
1. camera: code for camera calibration, (with and without resize)
2. down_camera_of: OpticalFlow Method; creating sequence for downward facing camera from zed
3. front_camera_vo: FrontFacingVO; creating sequence for forward facing camera from zed along with depth images
4. multicam: creating sequence for 3 camera streams.