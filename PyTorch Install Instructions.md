# PyTorch Installation instructions

https://pytorch.org
Download through Quick Start Locally option.
Select these conditions[Depends on your machine]: Stable, Windows, Pip, Python 3.5, 9.0 
You will recieve this command:
  pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp35-cp35m-win_amd64.whl
  pip3 install torchvision
  OR
  pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-win_amd64.whl
  pip install torchvision
  
# Installing CUDA so that i can use NVidea CUDA cores to do computation.
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
 
Now, the websites will say that you need visual studio and stuff....
actually maybe you do...
I think i had to install anaconda.
 
http://julip.co/2009/09/how-to-install-and-configure-cuda-on-windows/
I might have used steps 2 and 3 in the above link. I skipped 1. Visual studio i think.
Step 2 was to install the latest NVIDEA Driver. Makes sense...
Steo 3 was to install the CUDA Toolkit and SDK --> https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
I chose the options: Windows, x86_63, Version 10, exe local
 
https://developer.nvidia.com/how-to-cuda-python
This above link is how to use CUDA with Python. I believe that the video at the end helped alot too.
 
# Installing Anaconda because of a missing depencency i think. (maybe the visual studio crap)
https://www.anaconda.com/download/#windows

I opened the Anaconda Prompt to then run:
```
conda update conda
conda install cudatoolkit
jupyter Notebook
```

Not sure what this link was for --> Looks like i was trying to change the python version https://conda.io/docs/user-guide/tasks/manage-python.html

# Python Cuda code to run
https://github.com/siddharthsharmanv/cudacasts/blob/master/InstallingCUDAPython/VectorAdd.py

DID I DO THIS THO??? also from the pytorch website
conda install pytorch -c pytorch
pip3 install torchvision


MAYBE I NEVER INSTALLED TORCH WTFGHGHG


