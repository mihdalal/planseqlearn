FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04


RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y zsh libglfw3-dev libgles2-mesa-dev patchelf

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.8-dev python3.8 python3-pip
RUN apt-get install -y python-is-python3


ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN cp -R /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN mkdir /root/code
WORKDIR /root/code

RUN pip install mujoco_py
WORKDIR /mujoco_py

# For atari-py
RUN apt-get install -y zlib1g-dev swig cmake

RUN pip install glfw>=1.4.0
RUN pip install numpy>=1.11
RUN pip install Cython>=0.27.2
RUN pip install imageio>=2.1.2
RUN pip install cffi>=1.10
RUN pip install imagehash>=3.4
RUN pip install ipdb
RUN pip install Pillow>=4.0.0
RUN pip install pycparser>=2.17.0
RUN pip install pytest>=3.0.5
RUN pip install pytest-instafail
RUN pip install scipy>=0.18.0
RUN pip install sphinx
RUN pip install sphinx_rtd_theme
RUN pip install numpydoc
RUN pip install cloudpickle==1.3.0
RUN pip install cached-property==1.3.1
RUN pip install gitpython==2.1.7
RUN pip install gtimer
RUN pip install awscli
RUN pip install boto3
RUN pip install joblib
RUN pip install sk-video
RUN pip install torch torchvision
RUN pip install opencv-python==4.3.0.36
RUN pip install matplotlib
RUN pip install stable_baselines3
RUN pip install h5py
RUN pip install termcolor
RUN pip install click
RUN pip install numba
RUN pip install numpy-quaternion
RUN mkdir /usr/lib/nvidia-000/
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/lib/nvidia-000

RUN pip install numpy==1.22.4
RUN pip install absl-py==0.13.0
RUN pip install jupyterlab==3.0.14
RUN pip install scikit-image==0.18.1
RUN pip install termcolor==2.1.1
RUN pip install tb-nightly==2.11.0a20221013
RUN pip install imageio==2.19.3
RUN pip install imageio-ffmpeg==0.4.4
RUN pip install hydra-core==1.1.0
RUN pip install hydra-submitit-launcher==1.1.5
RUN pip install pandas==1.5.2
RUN pip install ipdb==0.13.9
RUN pip install yapf==0.31.0
RUN pip install sklearn==0.0
RUN pip install matplotlib==3.5.1
RUN pip install opencv-python==4.6.0.66
RUN pip install wandb==0.13.5
RUN pip install pylint==2.15.9
RUN pip install distracting_control==0.1.1rc3
RUN pip install six
RUN pip install git+https://github.com/aravindr93/mjrl@master#egg=mjrl
RUN pip install mujoco==2.3.5
RUN pip install gym
RUN pip install open3d
COPY mjkey.txt /root/.mujoco/mjkey.txt


RUN mkdir -p /root/.mujoco \
    && wget https://github.com/google-deepmind/mujoco/archive/refs/tags/2.1.0.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN cp -R /root/.mujoco/mujoco-2.1.0 /root/.mujoco/mujoco210/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210_linux/bin:${LD_LIBRARY_PATH}

# https://github.com/openai/mujoco-py/issues/773 -> mujoco py cythonization fails with cython 3.0
RUN pip install "cython<3" 
RUN pip install mujoco_py==2.0.2.13
RUN python -c 'import mujoco_py'

# add /mnt/mprl/doodad to PYTHONPATH
ENV PYTHONPATH /mnt/mprl/doodad:$PYTHONPATH
ENV PYTHONPATH /mnt/mprl/rlkit:$PYTHONPATH
ENV PYTHONPATH /mnt/mprl/robosuite:$PYTHONPATH
ENV PYTHONPATH /mnt/mprl/d4rl_latest:$PYTHONPATH

RUN pip uninstall -y gitdb2
RUN pip install gitdb2==2.0.6
RUN pip install pybullet-svl
RUN pip install plantcv
RUN pip install urdfpy trimesh
RUN pip install PyOpenGL==3.1.6
ENV PYTHONPATH /mnt/mprl/metaworld:$PYTHONPATH
ENV PYTHONPATH /mnt/mprl/mopa-rl-1:$PYTHONPATH
RUN pip install dm_control==1.0.5
COPY . /ompl
# RUN /ompl/install-ompl-ubuntu.sh --python
# RUN mv ompl-1.5.2/ /ompl/
