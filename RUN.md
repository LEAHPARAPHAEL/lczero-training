# proto
protoc -I=. --python_out=. proto/net.proto

# uv
UV_HTTP_TIMEOUT=600 uv pip install -r tf/requirements.txt --index-strategy unsafe-best-match --no-build-isolation


    uv run train.py --cfg configs/Cx2-Tx8.yaml

# docker
docker run --gpus all -it --rm --ipc=host --user $(id -u):$(id -g) -v ~/lczero-training:/workspace/lczero -v ~/leela:/home/raph/leela nvcr.io/nvidia/tensorflow:25.02-tf2-py3

cd lczero/tf

    python train.py --cfg configs/Cx2-Tx8.yaml

# git
git config --global pull.rebase true 
git status 
git add 
git rebase --continue