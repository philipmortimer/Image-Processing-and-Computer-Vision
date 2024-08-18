This readme contains instructions on how to run the source code. On the MVB lab machines, do the following.
1. Open a conda enabled terminal in this directory. (On lab machines this can be done by opening a terminal in this directory and running 'module load anaconda/3-2023').
2. Run 'conda create -n ipcv_part2_run python=3.8'
3. Run 'conda activate ipcv_part2_run'
4. Run 'pip install -r requirements.txt'
5. Run 'python CWII2324-v2.py'

This will produce visualisations of all parts of the task, including the original image, the Hough circle detection, the epipolar lines, the correspondence between the two images, the sphere centre reconstructions and the sphere reconstructions. Images for each part will also be saved within the 'Images' folder. Rememeber to run 'conda deactivate' once finished.

I developed my system on windows. I found that the following makes the system run on my windows laptop.
1. Open a conda enabled terminal in this directory. (In windows this involves opening command prompt in this directory).
2. Run 'conda create -n ipcv_part2_run python=3.8'
3. Run 'conda activate ipcv_part2_run'
4. Run 'pip install -r windowsrequirements.txt'
5. Run 'python CWII2324-v2.py'
