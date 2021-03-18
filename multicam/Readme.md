# File Utils for Multicam image splitting.

For this, the input is a folder of images of the format : camx_timestamp.jpeg where timestamp is a float; eg: cam2_13234.234.jpeg.
Cam1 is for center cam
Cam2 is for left cam (for downfacing camera maybe)
Cam3 is for right cam (for right camera)

First run display_multicam_imgs to run through the images, and when the start or end of sequence is encountered, print the file index.
Once the starts and ends are identified, fill them up in the write_multicam_sequence script, and run the script.