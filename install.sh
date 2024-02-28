# #! /bin/sh
# git lfs install
# git lfs track "*.plugin"
# mamba create -n psl python=3.8
# mkdir -p ~/.mujoco
# wget https://www.roboti.us/download/mujoco200_linux.zip -O ~/.mujoco/mujoco.zip
# unzip ~/.mujoco/mujoco.zip -d ~/.mujoco/
# rm ~/.mujoco/mujoco.zip
eval "$(conda shell.bash hook)"
conda activate psl
pip install -e d4rl/
pip install -e mjrl/
pip install -e metaworld/
pip install robosuite/
pip install -e doodad/
pip install -e mopa-rl/
pip install -e rlkit/
pip install -r requirements.txt
pip install dm-env
pip install distracting-control
pip install --upgrade dm-control
pip install mujoco==2.3.5
pip install numpy==1.23.5
pip install mujoco-py==2.0.2.5
pip install --upgrade networkx # for removing annoying warning
unzip containers/ompl-1.5.2.zip -d containers/
echo "containers/ompl-1.5.2/py-bindings" >> ~/mambaforge/envs/planseqlearn/lib/python3.8/site-packages/ompl.pth
pip install -e .