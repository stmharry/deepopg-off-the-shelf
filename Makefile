#################
### VARIABLES ###
#################

### paths

CONFIG_DIR ?= ./configs
DATA_DIR ?= $(ROOT_DIR)/data
MODEL_DIR_ROOT ?= $(ROOT_DIR)/models
RESULT_DIR_ROOT ?= $(ROOT_DIR)/results

MODEL_DIR ?= $(MODEL_DIR_ROOT)/$(MODEL_NAME)
RESULT_DIR ?= $(RESULT_DIR_ROOT)/$(RESULT_NAME)
CONFIG_FILE ?= $(CONFIG_DIR)/$(CONFIG_NAME)

LATEST_MODEL_CHECKPOINT ?= $(shell realpath --relative-to=$(MODEL_DIR) $(shell ls -t $(MODEL_DIR)/model_*.pth | head -n1))
YOLO_LATEST_MODEL_CHECKPOINT ?= weights/best.pt
NEW_NAME ?= $(shell date "+%Y-%m-%d-%H%M%S")
COCO_ANNOTATOR_URL ?= http://192.168.0.79:5000/api

### executables

PYTHONPATH ?= ./detectron2:./MaskDINO
PYTHON ?= python
PY ?= \
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	PYTHONPATH=$(PYTHONPATH):. \
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
		$(PYTHON)

# we enter yolo with a script to patch `amp`
YOLO_TRAIN ?= $(PY) scripts/main_yolo.py segment train
YOLO_PREDICT ?= \
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(shell which yolo) segment predict

### arguments

MAIN = scripts/main.py \
	--main-app $(MAIN_APP) \
	--config-file $(CONFIG_FILE) \
	--data-dir $(DATA_DIR)

COMMON_ARGS = \
	--data_dir $(DATA_DIR) \
	--result_dir $(RESULT_DIR) \
	--dataset_name $(DATASET_NAME)

### variables

MIN_SCORE ?= 0.0001
MIN_IOU ?= 0.0
MAX_OBJS ?= 500
PREDICTION_NAME ?= instances_predictions.pth
CSV_NAME ?= result.csv
VISUALIZE_DIR ?= $(subst instances_predictions,visualize,$(basename $(PREDICTION_NAME)))
CPUS ?= $(shell echo $$(( $(shell nproc --all) - 2 )))

ifeq ($(CUDA_VISIBLE_DEVICES),)
	DEVICE = cpu
else
	DEVICE = cuda
endif

ifeq ($(ARCH),maskdino)
	MAIN_APP = train_net:main
else ifeq ($(ARCH),mvitv2)
	MAIN_APP = tools.lazyconfig_train_net:main
else ifeq ($(ARCH),deeplab)
	MAIN_APP = projects.DeepLab.train_net:main
endif

###############
### TARGETS ###
###############

default:
	echo $(LATEST_MODEL)

### util targets

check-%:
	@if [ -z '${${*}}' ]; then echo 'Environment variable $* not set' && exit 1; fi

--check-MAIN: check-ROOT_DIR check-MAIN_APP check-CONFIG_NAME
--check-COMMON: check-ROOT_DIR check-RESULT_NAME check-DATASET_NAME
--check-COCO: check-COCO_ANNOTATOR_USERNAME check-COCO_ANNOTATOR_PASSWORD

### data preprocessing targets

convert-ntuh-coco-golden-label: ROOT_DIR = $(RAW_DIR)
convert-ntuh-coco-golden-label: check-RAW_DIR
convert-ntuh-coco-golden-label:
	$(PY) scripts/$@.py \
		--input $(DATA_DIR)/raw/NTUH/ntuh-opg-12.json \
		--output $(DATA_DIR)/coco/instance-detection-v1-ntuh.json

convert-ntuh-finding-golden-label: ROOT_DIR = $(RAW_DIR)
convert-ntuh-finding-golden-label: check-RAW_DIR
convert-ntuh-finding-golden-label:
	$(PY) scripts/$@.py \
		--input "$(DATA_DIR)/raw/NTUH/golden_label/(WIP) NTUH Summary Golden Label - Per-study.csv" \
		--input_coco $(DATA_DIR)/raw/NTUH/ntuh-opg-12.json \
		--output $(DATA_DIR)/csvs/pano_ntuh_golden_label.csv

convert-ntuh-finding-human-label: ROOT_DIR = $(RAW_DIR)
convert-ntuh-finding-human-label: check-RAW_DIR
convert-ntuh-finding-human-label:
	$(PY) scripts/$@.py \
		--input_dir $(DATA_DIR)/raw/NTUH/human_label \
		--output "$(DATA_DIR)/csvs/pano_ntuh_human_label_{}.csv"

convert-coco-to-detectron2-semseg: PYTHONPATH = ./detectron2
convert-coco-to-detectron2-semseg: ROOT_DIR = $(RAW_DIR)
convert-coco-to-detectron2-semseg: check-RAW_DIR
convert-coco-to-detectron2-semseg:
	$(PY) scripts/$@.py \
		--data_dir $(DATA_DIR) \
		--dataset_name $(DATASET_NAME) \
		--mask_dir "masks/segmentation-v4" \
		--num_processes $(CPUS)

### maskdino targets

install-maskdino:
	@$(PY) ./MaskDINO/maskdino/modeling/pixel_decoder/ops/setup.py build install

train-maskdino: MODEL_NAME ?= $(NEW_NAME)
train-maskdino: CONFIG_NAME ?= config-maskdino-r50.yaml
train-maskdino: --check-MAIN
train-maskdino:
	$(PY) $(MAIN) \
		OUTPUT_DIR $(MODEL_DIR)

test-maskdino: RESULT_NAME ?= $(NEW_NAME)
test-maskdino: CONFIG_FILE ?= $(MODEL_DIR)/config.yaml
test-maskdino: MODEL_CHECKPOINT ?= $(LATEST_MODEL_CHECKPOINT)
test-maskdino: --check-MAIN check-MODEL_NAME
test-maskdino:
	$(PY) $(MAIN) --eval-only \
		MODEL.WEIGHTS $(MODEL_DIR)/$(MODEL_CHECKPOINT) \
		DATASETS.TEST "('pano_eval',)"

debug-maskdino: PYTHON = python -m pdb
debug-maskdino: MODEL_DIR = /tmp/debug
debug-maskdino: --check-MAIN
debug-maskdino:
	$(PY) $(MAIN) \
		OUTPUT_DIR $(MODEL_DIR)
		DATASETS.TEST "('pano_debug',)" \
		TEST.EVAL_PERIOD 10

### detectron2 targets

install-detectron2:
	@pip install -e ./detectron2 && \
		ln -sf ../../configs ./detectron2/detectron2/model_zoo/

# mvitv2 targets

train-mvitv2: MODEL_NAME ?= $(NEW_NAME)
train-mvitv2: CONFIG_NAME ?= mask_rcnn_mvitv2_t_3x.py
train-mvitv2: --check-MAIN
train-mvitv2:
	$(PY) $(MAIN) \
		train.output_dir=$(MODEL_DIR)

test-mvitv2: RESULT_NAME ?= $(NEW_NAME)
test-mvitv2: CONFIG_FILE ?= $(MODEL_DIR)/config.yaml
test-mvitv2: MODEL_CHECKPOINT ?= $(LATEST_MODEL_CHECKPOINT)
test-mvitv2: --check-MAIN check-MODEL_NAME
test-mvitv2:
	$(PY) $(MAIN) --eval-only \
		train.init_checkpoint=$(MODEL_DIR)/$(MODEL_CHECKPOINT) \
		train.output_dir=$(RESULT_DIR) \
		dataloader.test.dataset.names=$(DATASET_NAME) \
		dataloader.test.dataset.filter_empty=False \
		dataloader.evaluator.output_dir=$(RESULT_DIR) \
		model.roi_heads.box_predictor.test_score_thresh=$(MIN_SCORE) \
		model.roi_heads.box_predictor.test_nms_thresh=$(MIN_IOU) \
		model.roi_heads.box_predictor.test_topk_per_image=$(MAX_OBJS)

debug-mvitv2: PYTHON = python -m pdb
debug-mvitv2: MODEL_DIR = /tmp/debug
debug-mvitv2: CONFIG_NAME = mask_rcnn_mvitv2_t_3x.py
debug-mvitv2: --check-MAIN
debug-mvitv2:
	$(PY) $(MAIN) \
		train.output_dir=$(MODEL_DIR) \
		dataloader.train.dataset.names=pano_debug

train-deeplab: MODEL_NAME ?= $(NEW_NAME)
train-deeplab: CONFIG_NAME ?= deeplab-v3.yaml
train-deeplab: --check-MAIN
train-deeplab:
	$(PY) $(MAIN) \
		OUTPUT_DIR $(MODEL_DIR)

test-deeplab: RESULT_NAME ?= $(NEW_NAME)
test-deeplab: CONFIG_FILE = $(MODEL_DIR)/config.yaml
test-deeplab: MODEL_CHECKPOINT ?= $(LATEST_MODEL_CHECKPOINT)
test-deeplab: check-ROOT_DIR check-MAIN_APP check-MODEL_NAME
test-deeplab:
	$(PY) $(MAIN) --eval-only \
		OUTPUT_DIR $(RESULT_DIR) \
		MODEL.DEVICE $(DEVICE) \
		DATASETS.TEST "('$(DATASET_NAME)',)" \
		MODEL.WEIGHTS $(MODEL_DIR)/$(MODEL_CHECKPOINT) \
		INPUT.CROP.ENABLED False

debug-deeplab: PYTHON = python -m pdb
debug-deeplab: MODEL_DIR = /tmp/debug
debug-deeplab: CONFIG_NAME = deeplab-v3.yaml
debug-deeplab: --check-MAIN
debug-deeplab:
	$(PY) $(MAIN) \
		DATALOADER.NUM_WORKERS \
		OUTPUT_DIR $(MODEL_DIR)

### yolo targets

convert-coco-to-yolo: check-ROOT_DIR check-DATASET_NAME
convert-coco-to-yolo:
	$(PY) scripts/$@.py \
		--data_dir $(DATA_DIR) \
		--dataset_name $(DATASET_NAME)

convert-yolo-labels-to-detectron2-prediction-pt: --check-COMMON check-PREDICTION_NAME
convert-yolo-labels-to-detectron2-prediction-pt:
	$(PY) scripts/$@.py $(COMMON_ARGS) \
		--prediction_name $(PREDICTION_NAME)

# when passing `cfg`, all other arguments will be ignored,
# so we dump the config to a temp file and append the rest
train-yolo: MODEL_NAME = $(NEW_NAME)
train-yolo: CONFIG_NAME ?= yolov8n-seg.yaml
train-yolo: TMP_FILE := $(shell mktemp --suffix=.yaml)
train-yolo: check-ROOT_DIR
train-yolo:
	cat $(CONFIG_FILE) > $(TMP_FILE) && \
		echo "mode: train" >> $(TMP_FILE) && \
		echo "data: $(DATA_DIR)/yolo/metadata.yaml" >> $(TMP_FILE) && \
		echo "project: $(MODEL_DIR_ROOT)" >> $(TMP_FILE) && \
		echo "name: ./$(MODEL_NAME)" >> $(TMP_FILE) && \
		$(YOLO_TRAIN) cfg="$(TMP_FILE)"

test-yolo: RESULT_NAME ?= $(NEW_NAME)
test-yolo: MODEL_CHECKPOINT ?= $(YOLO_LATEST_MODEL_CHECKPOINT)
test-yolo: check-ROOT_DIR check-RESULT_NAME check-DATASET_NAME
test-yolo:
	$(YOLO_PREDICT) \
		source="$(DATA_DIR)/yolo/$(DATASET_NAME).txt" \
		project="$(RESULT_DIR_ROOT)" \
		name="./$(RESULT_NAME)" \
		model="$(MODEL_DIR)/$(MODEL_CHECKPOINT)" \
		exist_ok=True \
		conf=$(MIN_SCORE) \
		iou=$(MIN_IOU) \
		max_det=$(MAX_OBJS) \
		save=True \
		save_txt=True \
		save_conf=True \
		save_crop=True

### overall targets

install: install-maskdino install-detectron2
train: check-ARCH train-$(ARCH)
test: check-ARCH test-$(ARCH)
debug: check-ARCH debug-$(ARCH)

coco-annotator:
	cd coco-annotator && \
		docker compose up --build --detach

--postprocess: --check-COMMON
--postprocess:
	$(PY) scripts/postprocess.py $(COMMON_ARGS) \
		--use_gt_as_prediction $(USE_GT) \
		--input_prediction_name $(PREDICTION_NAME) \
		--output_prediction_name $(OUTPUT_PREDICTION_NAME) \
		--csv_name $(CSV_NAME) \
		--min_score $(MIN_SCORE)

postprocess: USE_GT = false
postprocess: OUTPUT_PREDICTION_NAME = $(PREDICTION_NAME:.pth=.postprocessed.pth)
postprocess: --postprocess

postprocess-gt: USE_GT = true
postprocess-gt: OUTPUT_PREDICTION_NAME = instances_predictions.pth
postprocess-gt: --postprocess

--visualize: --check-COMMON
--visualize:
	$(PY) scripts/visualize.py $(COMMON_ARGS) \
		--use_gt_as_prediction $(USE_GT) \
		--prediction_name $(PREDICTION_NAME) \
		--visualize_dir $(VISUALIZE_DIR)

visualize: USE_GT = false
visualize: --visualize

visualize-gt: USE_GT = true
visualize-gt: --visualize

visualize-coco: --check-COMMON --check-COCO
visualize-coco:
	$(PY) scripts/$@.py $(COMMON_ARGS) \
		--prediction_name $(PREDICTION_NAME) \
		--coco_annotator_url $(COCO_ANNOTATOR_URL)

evaluate-auroc: check-ROOT_DIR
evaluate-auroc:
	$(PY) scripts/$@.py \
		--result_dir $(RESULT_DIR) \
		--csv_name $(CSV_NAME) \
		--golden_csv_path $(DATA_DIR)/csvs/pano_ntuh_golden_label.csv \
		--false_negative_csv_name false-negative.csv
