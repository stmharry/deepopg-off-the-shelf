PYTHONPATH = ./MaskDINO

PY = PYTHONPATH=$(PYTHONPATH) python
MAIN = main.py --config-file configs/config.yaml

OUTPUT_DIR = ./output
LATEST_MODEL = $(shell ls -t $(OUTPUT_DIR)/model_*.pth | head -n1)

setup:
	$(PY) MaskDINO/maskdino/modeling/pixel_decoder/ops/setup.py build install

train: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
train:
	$(PY) $(MAIN)

test:
	$(PY) $(MAIN) \
		--eval-only \
		MODEL.WEIGHTS ./output/model_0099999.pth \
		DATASETS.TEST "('pano_debug',)"

visualize:
	$(PY) visualize.py

debug:
	$(PY) $(MAIN) \
		OUTPUT_DIR ./debug \
		DATASETS.TRAIN "('pano_debug',)"

clean:
	-rm -rf $(OUTPUT_DIR)
