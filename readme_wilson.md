
# develop in container
docker run --shm-size=16g --privileged -it --gpus all  -v /home/wilsxue/APA:/workspace/APA  --name gcn_wilson vad_torch1.12_x86_v1.0 /bin/bash

# develop in container for export onnx 
docker run --shm-size=16g --privileged -it --gpus '"device=1"'  -v /home/wilsxue/APA:/workspace/APA  --name gcn_onnx gcn_docker_conda /bin/bash

# create a anaconda virtual environment for python3.6
conda create -n gcn_slots python=3.10.0

# activate the virtual environment
source activate gcn_slots

# install pytorch via pip
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# pin install other packages numpy numba tensorboardX easydict pyyaml permissive_dict opencv-python==4.0.0.21 scipy tqdm onnx==1.6.0

# run inference
export PYTHONPATH=`pwd`
python tools/demo.py -c config/ps_gat.yaml -m checkpoint_epoch_200.pth

# export ot onnx 
python3 tools/export_onnx.py -c config/ps_gat.yaml -m checkpoint_epoch_200.pth 


# convert onnx to tensorrt version , and the tensorrt has been installed with 8.6.1.6 
