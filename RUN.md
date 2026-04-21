# uv
UV_HTTP_TIMEOUT=600 uv pip install -r tf/requirements.txt --index-strategy unsafe-best-match --no-build-isolation

    uv run train.py --cfg configs/8x256x8h.yaml --output /tmp/8x256x8h.txt

    uv run train.py --cfg configs/8x256x8h-crbk.yaml --output /tmp/8x256x8h-crbk.txt

# docker
docker run --gpus all -it --rm --ipc=host
-v ~/lczero-training:/workspace/lczero
-v ~/leela:/home/raph/leela
nvcr.io/nvidia/tensorflow:25.02-tf2-py3

cd lczero/tf

    python train.py --cfg configs/8x256x8h.yaml --output /tmp/8x256x8h.txt

    python train.py --cfg configs/8x256x8h-crbk.yaml --output /tmp/8x256x8h-crbk.txt

# git
git config --global pull.rebase true git status git add git rebase --continue