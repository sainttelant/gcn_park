
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
# try to using trtexec to convert
# here is fixed input shape
trtexec --onnx=cache/ps_gat/100/output_onnx/fixed_model.onnx \
        --dumpLayerInfo \
        --saveEngine=cache/ps_gat/100/output_onnx/new.engine \
        --verbose >> trtexec_log.txt

# here using dynamic input shape
trtexec --onnx=cache/ps_gat/100/output_onnx/model.onnx \
        --dumpLayerInfo \
        --saveEngine=cache/ps_gat/100/output_onnx/new.engine \
        --verbose \
        --minShapes=image:1x3x256x256 \
        --optShapes=image:1x3x512x512 \
        --maxShapes=image:1x3x1024x1024 \
        >> trtexec_log.txt

# trtexec convert gnn onnx 
trtexec --onnx=cache/ps_gat/100/output_onnx/gnn_model.onnx \
        --dumpLayerInfo \
        --saveEngine=cache/ps_gat/100/output_onnx/gnn_new.engine \
        --verbose \
        --minShapes=descriptors:1x128x10,points:1x10x2  \
        --optShapes=descriptors:1x128x50,points:1x50x2 \
        --maxShapes=descriptors:1x128x100,points:1x100x2 \
        >> trtexec_gnn_log.txt


