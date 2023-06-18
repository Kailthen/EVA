
# server

## install
```
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH}
PY3D="python3 -m debugpy --wait-for-client --listen 0.0.0.0:5678"
PY3D="python3"

# install

conda create -n eva01 python==3.8
conda activate eva01

# torch 2.0.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# build EVA det / Detectron2 from source
cd EVA-01/det
pip install -e .
```

## demo 
```
cd EVA-01/det
${PY3D} serv/serv.py \
  --fp16 \
#  --output ./tmp \
  --input ./serv/000154.jpg \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py \
  --opts train.init_checkpoint=../modelzoo/eva_coco_det.pth
```

## start server
```
cd EVA-01/det
python serv/web_serv.py \
  --fp16 \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py \
  --opts train.init_checkpoint=../modelzoo/eva_coco_det.pth
```

# client
```
cd EVA-01/det
python serv/web_serv_client.py \
    --input ./demo/000002.png \
    --output_dir ./demo/ \
    --vis \
    --host http://127.0.0.1:8801
```
