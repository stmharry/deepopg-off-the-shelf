PYTHONPATH = ./MaskDINO:.
PRETRAINED_MODEL_DIR = /mnt/hdd/PANO/models/pretrained

PY = PYTHONPATH=$(PYTHONPATH) python

setup:
	$(PY) MaskDINO/maskdino/modeling/pixel_decoder/ops/setup.py build install

train: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
train:
	$(PY) main.py \
		--config-file configs/config.yaml
