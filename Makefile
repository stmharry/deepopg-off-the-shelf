### variables

# common variables

ROOT_DIR = /mnt/hdd/PANO
DATA_DIR = $(ROOT_DIR)/data
MODEL_DIR_ROOT = $(ROOT_DIR)/models
RESULT_DIR_ROOT = $(ROOT_DIR)/results

# default variables

PYTHONPATH = .
PYTHON = python
PY = \
	PYTHONPATH=$(PYTHONPATH):. \
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
		$(PYTHON)

MAIN = scripts/main.py \
	--main-app $(MAIN_APP) \
	--config-file $(CONFIG_FILE) \
	--data-dir $(DATA_DIR)

COMMANDS = scripts/commands.py \
	--data_dir $(DATA_DIR) \
	--result_dir $(RESULT_DIR) \
	--dataset_name $(DATASET_NAME) \

# functions

NEW_NAME = $(shell date "+%Y-%m-%d-%H%M%S")

MODEL_DIR = $(MODEL_DIR_ROOT)/$(MODEL_NAME)
LATEST_MODEL = $(shell ls -t $(MODEL_DIR)/model_*.pth | head -n1)

RESULT_DIR = $(RESULT_DIR_ROOT)/$(RESULT_NAME)

CONFIG_DIR = ./configs
CONFIG_FILE = $(CONFIG_DIR)/$(CONFIG_NAME)

ifeq ($(ARCH),maskdino)
	PYTHONPATH = ./MaskDINO
	MAIN_APP = train_net:main

else ifeq ($(ARCH),detectron2)
	PYTHONPATH = ./detectron2
	MAIN_APP = tools.lazyconfig_train_net:main

endif

### targets

# utils

check-%:
	@if [ -z '${${*}}' ]; then echo 'Environment variable $* not set' && exit 1; fi

# maskdino targets

install-maskdino:
	@$(PY) ./MaskDINO/maskdino/modeling/pixel_decoder/ops/install.py build install

train-maskdino: MODEL_NAME = $(NEW_NAME)
train-maskdino: CONFIG_NAME = config-maskdino-r50.yaml
train-maskdino:
	$(PY) $(MAIN) \
		OUTPUT_DIR $(MODEL_DIR)

test-maskdino: RESULT_NAME = $(NEW_NAME)
test-maskdino: CONFIG_FILE = $(MODEL_DIR)/config.yaml
test-maskdino: check-MODEL_NAME
	$(PY) $(MAIN) --eval-only \
		MODEL.WEIGHTS $(LATEST_MODEL) \
		DATASETS.TEST "('pano_eval',)"

debug-maskdino: PYTHON = python -m pdb
debug-maskdino: MODEL_DIR = /tmp/debug
debug-maskdino:
	$(PY) $(MAIN) \
		OUTPUT_DIR $(MODEL_DIR)
		DATASETS.TEST "('pano_debug',)" \
		TEST.EVAL_PERIOD 10

# detectron2 target

install-detectron2:
	@pip install -e ./detectron2 && \
		ln -sf ../../configs ./detectron2/detectron2/model_zoo/

train-detectron2: MODEL_NAME = $(NEW_NAME)
train-detectron2: CONFIG_NAME = mask_rcnn_mvitv2_t_3x.py
train-detectron2:
	$(PY) $(MAIN) \
		train.output_dir=$(MODEL_DIR)

test-detectron2: RESULT_NAME = $(NEW_NAME)
test-detectron2: CONFIG_FILE = $(MODEL_DIR)/config.yaml
test-detectron2: check-MODEL_NAME
	$(PY) $(MAIN) --eval-only \
		train.init_checkpoint=$(LATEST_MODEL) \
		train.output_dir=$(RESULT_DIR) \
		dataloader.test.dataset.names=$(DATASET_NAME) \
		dataloader.evaluator.output_dir=$(RESULT_DIR) \
		model.roi_heads.box_predictor.test_score_thresh=0.0 \
		model.roi_heads.box_predictor.test_nms_thresh=0.0 \
		model.roi_heads.box_predictor.test_topk_per_image=500

debug-detectron2: PYTHON = python -m pdb
debug-detectron2: MODEL_DIR = /tmp/debug
debug-detectron2:
	$(PY) $(MAIN) \
		train.output_dir=$(MODEL_DIR) \
		dataloader.train.dataset.names=pano_debug

# overall targets

install: install-maskdino install-detectron2
train: check-ARCH train-$(ARCH)
test: check-ARCH test-$(ARCH)
debug: check-ARCH debug-$(ARCH)

coco-annotator:
	cd coco-annotator && \
		docker compose up --build --detach

postprocess: check-DATASET_NAME check-RESULT_NAME
postprocess:
	$(PY) $(COMMANDS) \
		--do_postprocess \
		--output_prediction_name instances_predictions.postprocessed.pth

visualize: check-DATASET_NAME check-RESULT_NAME check-COCO_ANNOTATOR_USERNAME check-COCO_ANNOTATOR_PASSWORD
visualize:
	$(PY) $(COMMANDS) \
		--prediction_name instances_predictions.postprocessed.pth \
		--do_visualize \
		--visualizer_dir visualize.postprocessed \
		--nodo_coco \
		--coco_annotator_url http://192.168.0.79:5000/api
