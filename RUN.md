# proto
protoc -I=. --python_out=. proto/net.proto

# uv
UV_HTTP_TIMEOUT=600 uv pip install -r tf/requirements.txt --index-strategy unsafe-best-match --no-build-isolation


    uv run train.py --cfg configs/Cx2-Tx8.yaml

    uv run train.py --cfg configs/Tx8-eps-5.yaml

# docker
docker run --gpus all -it --rm --ipc=host --user $(id -u):$(id -g) -v ~/lczero-training:/workspace/lczero -v ~/leela:/home/raph/leela nvcr.io/nvidia/tensorflow:25.02-tf2-py3

docker run --gpus all -it --rm --ipc=host \
  --user $(id -u):$(id -g) \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/lczero-training:/workspace/lczero \
  -v ~/leela:/home/raph/leela \
  tensorflow-leela-gui

cd lczero/tf

    python train.py --cfg configs/Cx2-Tx8.yaml

# git
git config --global pull.rebase true 
git status 
git add 
git rebase --continue



uv run train.py --cfg configs/Tx8.yaml


# Onyxia
k7yvat5cbtbujdxazuif

# Jean-Zay

echo $SCRATCH
/lustre/fsn1/projects/rech/kwf/uzr96yg
/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data

echo $WORK
/lustre/fswork/projects/rech/kwf/uzr96yg


sbatch --job-name=lustre_purge --time=03:00:00 --ntasks=1 --cpus-per-task=1 --wrap="find -P data_to_delete \( -type f -o -type l \) -delete && find -P data_to_delete -type d -empty -delete" --partition=prepost --account=kwf@v100


# Local -> Jean-Zay
rsync -avzP leela/ jean-zay:/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/

# Jean-Zay -> Local
rsync -avzP jean-zay:/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/networks/ /home/raph/leela/networks/
rsync -avzP jean-zay:/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/logs/ /home/raph/leela/logs/

rsync -avzP jean-zay:/lustre/fswork/projects/rech/kwf/uzr96yg/lc0/logs/ /home/raph/leela/configs/