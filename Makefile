#################
### VARIABLES ###
#################

### paths

DATA_DIR ?= $(ROOT_DIR)/data

CONFIG_DIR ?= ./configs
CONFIG_FILE ?= $(CONFIG_DIR)/$(CONFIG_NAME)

MODEL_DIR_ROOT ?= $(ROOT_DIR)/models
MODEL_DIR ?= $(MODEL_DIR_ROOT)/$(MODEL_NAME)
MODEL_CONFIG_FILE ?= $(MODEL_DIR)/config.yaml

RESULT_DIR_ROOT ?= $(ROOT_DIR)/results
RESULT_DIR ?= $(RESULT_DIR_ROOT)/$(RESULT_NAME)

LATEST_MODEL_DETECTRON2 ?= $(shell realpath --relative-to=$(MODEL_DIR) $(shell ls -t $(MODEL_DIR)/model_*.pth | head -n1))
LATEST_MODEL_YOLO ?= weights/best.pt

NEW_NAME ?= $(shell date "+%Y-%m-%d-%H%M%S")

### executables

PYTHONPATH ?= ./detectron2:./MaskDINO
PYTHON ?= python
PY ?= \
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	PYTHONPATH=$(PYTHONPATH):. \
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
		$(PYTHON)

RUN_SCRIPT = $(PY) scripts/$(word 1,$(subst ., ,$(subst --,,$@))).py

# we enter yolo with a script to patch `amp`
YOLO_TRAIN ?= $(PY) scripts/main_yolo.py segment train
YOLO_PREDICT ?= \
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(shell which yolo) segment predict

### arguments

ifeq ($(ARCH),maskdino)
	MAIN_APP = train_net:main
else ifeq ($(ARCH),mvitv2)
	MAIN_APP = tools.lazyconfig_train_net:main
else ifeq ($(ARCH),deeplab)
	MAIN_APP = projects.DeepLab.train_net:main
endif

RUN_MAIN_DETECTRON2 = $(PY) scripts/main.py \
	--main-app $(MAIN_APP) \
	--config-file $(CONFIG_FILE) \
	--data-dir $(DATA_DIR) \
	--dataset_name $(DATASET_NAME)

COMMON_ARGS = \
	--data_dir $(DATA_DIR) \
	--result_dir $(RESULT_DIR) \
 	--dataset_name $(DATASET_NAME) \
 	--verbosity $(VERBOSITY)

### variables

MIN_SCORE ?= 0.0001
MIN_IOU ?= 0.5
MAX_OBJS ?= 300

YOLO_DIR ?= yolo
RESULT_CSV ?= result.csv
EVALUATION_DIR ?= $(subst result,evaluation,$(basename $(RESULT_CSV)))

PREDICTION ?= instances_predictions.pth
OUTPUT_PREDICTION ?= $(PREDICTION:.pth=.postprocessed.pth)
VISUALIZE_DIR ?= $(subst instances_predictions,visualize,$(basename $(PREDICTION)))

SEMSEG_PREDICTION ?= inference/sem_seg_predictions.json

VERBOSITY ?= 0
CPUS ?= $(shell echo $$(( $(shell nproc --all) - 2 )))

ifeq ($(CUDA_VISIBLE_DEVICES),)
	DEVICE = cpu
else
	DEVICE = cuda
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
--check-COCO: check-COCO_ANNOTATOR_URL check-COCO_ANNOTATOR_USERNAME check-COCO_ANNOTATOR_PASSWORD

### data preprocessing targets

convert-ntuh-coco-golden-label: ROOT_DIR = $(RAW_DIR)
convert-ntuh-coco-golden-label: check-RAW_DIR
convert-ntuh-coco-golden-label:
	$(RUN_SCRIPT) \
		--verbosity $(VERBOSITY) \
		--coco "$(DATA_DIR)/raw/NTUH/ntuh-opg-12.json" \
		--output_coco "$(DATA_DIR)/coco/instance-detection-v1-ntuh.json"

convert-ntuh-finding-golden-label: ROOT_DIR = $(RAW_DIR)
convert-ntuh-finding-golden-label: check-RAW_DIR
convert-ntuh-finding-golden-label:
	$(RUN_SCRIPT) \
		--verbosity $(VERBOSITY) \
		--label_csv "$(DATA_DIR)/raw/NTUH/golden_label/(WIP) NTUH Summary Golden Label - Per-study.csv" \
		--coco "$(DATA_DIR)/raw/NTUH/ntuh-opg-12.json" \
		--output_csv "$(DATA_DIR)/csvs/pano_ntuh_golden_label.csv"

convert-ntuh-finding-human-label: ROOT_DIR = $(RAW_DIR)
convert-ntuh-finding-human-label: check-RAW_DIR
convert-ntuh-finding-human-label:
	$(RUN_SCRIPT) \
		--verbosity $(VERBOSITY) \
		--label_dir "$(DATA_DIR)/raw/NTUH/human_label" \
		--output_csv "$(DATA_DIR)/csvs/pano_ntuh_human_label_{}.csv"

convert-coco-to-detectron2-semseg: ROOT_DIR = $(RAW_DIR)
convert-coco-to-detectron2-semseg: check-RAW_DIR
convert-coco-to-detectron2-semseg:
	$(RUN_SCRIPT) \
		--verbosity $(VERBOSITY) \
		--data_dir "$(DATA_DIR)" \
		--dataset_name $(DATASET_NAME) \
		--mask_dir "masks/segmentation-v4" \
		--num_workers $(CPUS)

### maskdino targets

install-maskdino:
	@$(PY) ./MaskDINO/maskdino/modeling/pixel_decoder/ops/setup.py build install

train-maskdino: MODEL_NAME ?= $(NEW_NAME)
train-maskdino: CONFIG_NAME ?= config-maskdino-r50.yaml
train-maskdino: DATASET_NAME = pano_train,pano_eval
train-maskdino: --check-MAIN
train-maskdino:
	$(RUN_MAIN_DETECTRON2) \
		OUTPUT_DIR $(MODEL_DIR)

test-maskdino: RESULT_NAME ?= $(NEW_NAME)
test-maskdino: CONFIG_FILE = $(MODEL_CONFIG_FILE)
test-maskdino: MODEL_CHECKPOINT ?= $(LATEST_MODEL_DETECTRON2)
test-maskdino: --check-MAIN check-MODEL_NAME
test-maskdino:
	$(RUN_MAIN_DETECTRON2) \
		--eval-only \
		MODEL.WEIGHTS $(MODEL_DIR)/$(MODEL_CHECKPOINT) \
		DATASETS.TEST "('$(DATASET_NAME)',)"

debug-maskdino: PYTHON = python -m pdb
debug-maskdino: MODEL_DIR = /tmp/debug
debug-maskdino: DATASET_NAME = pano_train,pano_eval
debug-maskdino: --check-MAIN
debug-maskdino:
	$(RUN_MAIN_DETECTRON2) \
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
train-mvitv2: DATASET_NAME = pano_train,pano_eval
train-mvitv2: --check-MAIN
train-mvitv2:
	$(RUN_MAIN_DETECTRON2) \
		train.output_dir=$(MODEL_DIR)

test-mvitv2: RESULT_NAME ?= $(NEW_NAME)
test-mvitv2: CONFIG_FILE = $(MODEL_CONFIG_FILE)
test-mvitv2: MODEL_CHECKPOINT ?= $(LATEST_MODEL_DETECTRON2)
test-mvitv2: --check-MAIN check-MODEL_NAME
test-mvitv2:
	$(RUN_MAIN_DETECTRON2) \
		--eval-only \
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
debug-mvitv2: DATASET_NAME = pano_train,pano_eval
debug-mvitv2: --check-MAIN
debug-mvitv2:
	$(RUN_MAIN_DETECTRON2) \
		train.output_dir=$(MODEL_DIR) \
		dataloader.train.dataset.names=pano_debug

# deeplab targets

train-deeplab: MODEL_NAME ?= $(NEW_NAME)
train-deeplab: CONFIG_NAME ?= deeplab-v3.yaml
train-deeplab: DATASET_NAME = pano_semseg_v4_train,pano_semseg_v4_eval
train-deeplab: --check-MAIN
train-deeplab:
	$(RUN_MAIN_DETECTRON2) \
		OUTPUT_DIR $(MODEL_DIR)

test-deeplab: RESULT_NAME ?= $(NEW_NAME)
test-deeplab: CONFIG_FILE = $(MODEL_CONFIG_FILE)
test-deeplab: MODEL_CHECKPOINT ?= $(LATEST_MODEL_DETECTRON2)
test-deeplab: --check-MAIN check-MODEL_NAME
test-deeplab:
	$(RUN_MAIN_DETECTRON2) \
		--eval-only \
		OUTPUT_DIR $(RESULT_DIR) \
		DATASETS.TEST "('$(DATASET_NAME)',)" \
		MODEL.DEVICE $(DEVICE) \
		MODEL.WEIGHTS $(MODEL_DIR)/$(MODEL_CHECKPOINT) \
		INPUT.CROP.ENABLED False \
		SOLVER.IMS_PER_BATCH 1 \
		TEST.SAVE_PROB True

debug-deeplab: PYTHON = python -m pdb
debug-deeplab: MODEL_DIR = /tmp/debug
debug-deeplab: CONFIG_NAME = deeplab-v3.yaml
debug-deeplab: DATASET_NAME = pano_semseg_v4_train,pano_semseg_v4_eval
debug-deeplab: --check-MAIN
debug-deeplab:
	$(RUN_MAIN_DETECTRON2) \
		DATALOADER.NUM_WORKERS 0 \
		SOLVER.MAX_ITER 10 \
		OUTPUT_DIR $(MODEL_DIR)

### yolo targets

convert-coco-to-yolo: check-ROOT_DIR check-DATASET_NAME
convert-coco-to-yolo:
	$(RUN_SCRIPT) \
		--verbosity $(VERBOSITY) \
		--data_dir $(DATA_DIR) \
		--dataset_name $(DATASET_NAME) \
		--yolo_dir $(YOLO_DIR)

convert-yolo-labels-to-detectron2-prediction-pt: --check-COMMON check-PREDICTION
convert-yolo-labels-to-detectron2-prediction-pt:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--prediction $(PREDICTION)

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
test-yolo: MODEL_CHECKPOINT ?= $(LATEST_MODEL_YOLO)
test-yolo: check-ROOT_DIR check-RESULT_NAME check-DATASET_NAME
test-yolo:
	$(YOLO_PREDICT) \
		source="$(DATA_DIR)/$(YOLO_DIR)/$(DATASET_NAME).txt" \
		project="$(RESULT_DIR_ROOT)" \
		name="./$(RESULT_NAME)" \
		model="$(MODEL_DIR)/$(MODEL_CHECKPOINT)" \
		exist_ok=True \
		conf=$(MIN_SCORE) \
		iou=$(MIN_IOU) \
		max_det=$(MAX_OBJS) \
		save=True \
		save_txt=True \
		save_conf=True

### overall targets

install: install-maskdino install-detectron2
train: check-ARCH train-$(ARCH)
test: check-ARCH test-$(ARCH)
debug: check-ARCH debug-$(ARCH)

coco-annotator:
	cd coco-annotator && \
		docker compose up --build --detach

--postprocess: --check-COMMON check-SEMSEG_RESULT_NAME check-SEMSEG_DATASET_NAME check-SEMSEG_PREDICTION
--postprocess:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		$(GT_ARG) \
		--prediction $(PREDICTION) \
		--semseg_result_dir $(RESULT_DIR_ROOT)/$(SEMSEG_RESULT_NAME) \
		--semseg_dataset_name $(SEMSEG_DATASET_NAME) \
		--semseg_prediction $(SEMSEG_PREDICTION) \
		--output_prediction $(OUTPUT_PREDICTION) \
		--output_csv $(RESULT_CSV) \
		--min_score $(MIN_SCORE) \
		--min_area 0 \
		--min_iom 0.3 \
		--nosave_predictions \
		--num_workers $(CPUS)

postprocess: GT_ARG = --nouse_gt_as_prediction
postprocess: --postprocess

postprocess.gt: GT_ARG = --use_gt_as_prediction
postprocess.gt: --postprocess

--visualize: --check-COMMON
--visualize:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		$(GT_ARG) \
		--prediction $(PREDICTION) \
		--visualize_dir $(VISUALIZE_DIR) \
		--visualize_subset \
		--min_score $(MIN_SCORE) \
		--num_workers $(CPUS)

visualize: GT_ARG = --nouse_gt_as_prediction
visualize: --visualize

visualize.gt: GT_ARG = --use_gt_as_prediction
visualize.gt: --visualize

visualize-semseg: --check-COMMON
visualize-semseg: VISUALIZE_DIR = visualize
visualize-semseg:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--prediction $(SEMSEG_PREDICTION) \
		--visualize_dir $(VISUALIZE_DIR)

visualize-coco: --check-COMMON --check-COCO
visualize-coco:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--prediction $(PREDICTION) \
		--coco_annotator_url $(COCO_ANNOTATOR_URL)

evaluate-auroc: check-ROOT_DIR
evaluate-auroc:
	$(RUN_SCRIPT) \
		--result_dir $(RESULT_DIR) \
		--csv $(RESULT_CSV) \
		--golden_csv_path "$(DATA_DIR)/csvs/$(DATASET_NAME)_golden_label.csv" \
		--evaluation_dir $(EVALUATION_DIR) \
		--verbosity $(VERBOSITY)

evaluate-auroc.with-human: check-ROOT_DIR
evaluate-auroc.with-human:
	$(RUN_SCRIPT) \
		--result_dir $(RESULT_DIR) \
		--csv $(RESULT_CSV) \
		--golden_csv_path "$(DATA_DIR)/csvs/$(DATASET_NAME)_golden_label.csv" \
		--human_csv_path "$(DATA_DIR)/csvs/$(DATASET_NAME)_human_label_{}.csv" \
		--evaluation_dir $(EVALUATION_DIR) \
		--verbosity $(VERBOSITY)

compare: check-ROOT_DIR check-IMAGE_PATTERNS
	$(RUN_SCRIPT) \
		--root_dir $(ROOT_DIR) \
		--height 400 \
		--image_patterns $(IMAGE_PATTERNS) \
		--output_html_path results/index.html
