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
RUN mkdir /usr/lib/nvidia-000/
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/lib/nvidia-000
COPY mjkey.txt /root/.mujoco/mjkey.txt

RUN mkdir -p /root/.mujoco \
    && wget https://github.com/google-deepmind/mujoco/archive/refs/tags/2.1.0.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN cp -R /root/.mujoco/mujoco-2.1.0 /root/.mujoco/mujoco210/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210_linux/bin:${LD_LIBRARY_PATH}

# https://github.com/openai/mujoco-py/issues/773 -> mujoco py cythonization fails with cython 3.0
RUN pip install cython==0.29.34
RUN pip install numpy==1.23.5
RUN pip install mujoco_py==2.0.2.5
RUN python -c 'import mujoco_py'

ENV PYTHONPATH /mnt/planseqlearn/doodad:$PYTHONPATH
ENV PYTHONPATH /mnt/planseqlearn/rlkit:$PYTHONPATH
ENV PYTHONPATH /mnt/planseqlearn/robosuite:$PYTHONPATH
ENV PYTHONPATH /mnt/planseqlearn/d4rl_latest:$PYTHONPATH
ENV PYTHONPATH /mnt/planseqlearn/metaworld:$PYTHONPATH
ENV PYTHONPATH /mnt/planseqlearn/d4rl:$PYTHONPATH
ENV PYTHONPATH /mnt/planseqlearn/mjrl:$PYTHONPATH
ENV PYTHONPATH /mnt/planseqlearn/mopa-rl:$PYTHONPATH
ENV PYTHONPATH /mnt/planseqlearn/DisentangledRep4RL:$PYTHONPATH

# copying over my ompl version which definitely works
COPY ompl-1.5.2/ /ompl
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN pip install dm-env distracting-control \
    && pip install --upgrade dm-control \
    && pip install mujoco==2.3.5 \
    && pip install numpy==1.23.5 \
    && pip install --upgrade networkx

RUN echo "/ompl/py-bindings" >> /usr/local/lib/python3.8/dist-packages/ompl.pth
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libjpeg8-dev \
    zlib1g-dev \
    libpython3.8 \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    libegl1 \
    libglfw3-dev \
    libglfw3 \
    libgl1-mesa-glx \
    libfdk-aac-dev \
    libass-dev \
    libopus-dev \
    libtheora-dev \
    libvorbis-dev \
    libvpx-dev \
    libssl-dev \
    libboost-serialization-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libeigen3-dev \
    libode-dev \
    libyaml-cpp-dev \
    libboost-python-dev \
    libboost-numpy-dev