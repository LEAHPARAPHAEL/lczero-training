# 4x128-128_no_mask
uv run train.py --cfg configs/4x128-128_no_mask.yaml --output /tmp/4x128-128_no_mask.txt

# 8x256-256_no_mask
uv run train.py --cfg configs/8x256-256_no_mask.yaml --output /tmp/8x256-256_no_mask.txt

# docker
docker run --gpus all -it --rm --ipc=host \
  -v ~/lczero-training:/workspace/lczero \
  -v ~/leela:/home/raph/leela \
  nvcr.io/nvidia/tensorflow:25.02-tf2-py3

cd lzero/tf

python train.py --cfg configs/8x256-256_no_mask.yaml --output /tmp/8x256-256_no_mask.txt

python train.py --cfg configs/8x256-256_rbk.yaml --output /tmp/8x256-256_rbk.txt
