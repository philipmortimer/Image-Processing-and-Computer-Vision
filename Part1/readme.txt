To run the final detector on the lab machine follow these steps.
1. Open a terminal in this directory.
2. Load conda using 'module load anaconda/3-2023' if applicable
3. 'conda create -n finaldetectorenv python=3.8'
4. 'conda activate finaldetectorenv'
5. 'pip install ultralytics'
6. 'python finaldetector.py -name Dartboard/dart0.jpg'   (replace with alternative filename as required)

The detected file is called "detected.jpg".

This approach has been tested and works on the 2.11 lab machines. As ultralytics is a large package, disk quota issues may occur if file system is full of other files. I had to delete documents and found that the source code and conda installation (with ultralytics) took up slightly under 9Gb. Given that students have a disk quota of around 12 Gg, it's a tight fit but is possible.

I ran all of the code on my Windows laptop. For suitable architectures, follow the following setup instructions to run all of the code (including the final detector and all other scripts). Alternatively, use pip to install individual packages as required (such as ultralytics).

1. Install conda.
2. Open a terminal in this directory that supports conda.
2. "conda create -n finaldetectorenv python=3.8"
3. "conda activate finaldetectorenv"
4. "pip install -r requirements.txt"
5. "python finaldetector.py -name Dartboard/dart0.jpg"   (replace with alternative filename as required)

Remember to deactivate conda once finished ("conda deactivate"). Alternatively, don't use conda and just use with a python 3.8 environment.

It appears that to run all code sections, the following sequence of installs will do the trick (though results may vary on different systems).
1. "pip install ultralytics"
2. "pip install tensorflow"