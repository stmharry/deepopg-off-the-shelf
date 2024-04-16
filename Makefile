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

CUDA_VISIBLE_DEVICES ?= 0
PYTHONPATH ?= ./detectron2:./MaskDINO:./google-health/analysis
PYTHON ?= python
PY ?= \
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	PYTHONPATH=$(PYTHONPATH):. \
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
		$(PYTHON)

PIP ?= $(PYTHON) -m pip

# for `--script.postfix`, we will run `script` target
RUN_SCRIPT = $(PY) scripts/$(word 1,$(subst ., ,$(subst --,,$@))).py
DOT ?= dot

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

DEBUG ?= false
VERBOSITY ?= 0
CPUS ?= 4

MIN_SCORE ?= 0.0001
MIN_IOU ?= 0.7
MAX_OBJS ?= 500

YOLO_DIR ?= yolo

MISSING_SCORING_METHOD ?= SHARE_NOBG
FINDING_SCORING_METHOD ?= SCORE_MUL_SHARE_NOBG_NOMUL_MISSING
POSTFIX ?= .postprocessed-with-$(SEMSEG_RESULT_NAME).missing-scoring-$(MISSING_SCORING_METHOD).finding-scoring-$(FINDING_SCORING_METHOD)

RESULT_CSV ?= result$(POSTFIX).csv
EVALUATION_DIR ?= $(subst result,evaluation.$(DATASET_NAME),$(basename $(RESULT_CSV)))

RAW_PREDICTION ?= instances_predictions.pth
PREDICTION ?= $(RAW_PREDICTION:.pth=$(POSTFIX).pth)
VISUALIZE_DIR ?= $(subst instances_predictions,visualize,$(basename $(PREDICTION)))

SEMSEG_PREDICTION ?= inference/sem_seg_predictions.json

CPROFILE_OUT ?= profile.out

ifneq ($(DEBUG),false)
	CPUS = 0
	VERBOSITY = 1
endif

ifeq ($(DEBUG),pdb)
	PYTHON = python -m pdb
else ifeq ($(DEBUG),memray)
	PYTHON = python -m memray run
else ifeq ($(DEBUG),cProfile)
	PYTHON = python -m cProfile -o $(CPROFILE_OUT)
endif

ifeq ($(CUDA_VISIBLE_DEVICES),)
	DEVICE = cpu
else
	DEVICE = cuda
endif

###############
### TARGETS ###
###############

default:

### util targets

check-%:
	@if [ -z '${${*}}' ]; then echo 'Environment variable $* not set' && exit 1; fi

--check-MAIN: check-ROOT_DIR check-MAIN_APP check-CONFIG_FILE
--check-COMMON: check-ROOT_DIR check-RESULT_NAME check-DATASET_NAME check-VERBOSITY
--check-COCO: check-COCO_ANNOTATOR_URL check-COCO_ANNOTATOR_USERNAME check-COCO_ANNOTATOR_PASSWORD

prof2png: DOT_OUT ?= $(CPROFILE_OUT:.out=.png)
prof2png:
	$(PYTHON) -m gprof2dot \
		-f pstats	$(CPROFILE_OUT) | \
		$(DOT) -T png -o $(DOT_OUT)

snakeviz: HOST = 0.0.0.0
snakeviz: PORT ?= 1235
snakeviz:
	$(PYTHON) -m snakeviz \
		--hostname $(HOST)  \
		--port $(PORT) \
		--server \
		$(CPROFILE_OUT)

### data preprocessing targets

convert-promaton-to-coco: ROOT_DIR = $(RAW_DIR)
convert-promaton-to-coco: check-RAW_DIR
convert-promaton-to-coco:
	$(RUN_SCRIPT) \
		--data_dir "$(DATA_DIR)" \
		--output_coco "$(DATA_DIR)/coco/promaton.1.json" \
		--num_workers $(CPUS) \
		--verbosity $(VERBOSITY)

# annotation

convert-ntuh-coco-golden-label: ROOT_DIR = $(RAW_DIR)
convert-ntuh-coco-golden-label: check-RAW_DIR
convert-ntuh-coco-golden-label:
	$(RUN_SCRIPT) \
		--coco "$(DATA_DIR)/raw/NTUH/ntuh-opg-12.json" \
		--output_coco "$(DATA_DIR)/coco/instance-detection-v1-ntuh.json" \
		--verbosity $(VERBOSITY)

convert-ntuh-finding-golden-label: ROOT_DIR = $(RAW_DIR)
convert-ntuh-finding-golden-label: check-RAW_DIR
convert-ntuh-finding-golden-label:
	$(RUN_SCRIPT) \
		--label_csv "$(DATA_DIR)/raw/NTUH/golden_label/(WIP) NTUH Summary Golden Label - Per-study.csv" \
		--coco "$(DATA_DIR)/raw/NTUH/ntuh-opg-12.json" \
		--output_csv "$(DATA_DIR)/csvs/pano_ntuh_golden_label.v2.csv" \
		--verbosity $(VERBOSITY)

convert-ntuh-finding-human-label: ROOT_DIR = $(RAW_DIR)
convert-ntuh-finding-human-label: check-RAW_DIR
convert-ntuh-finding-human-label:
	$(RUN_SCRIPT) \
		--label_dir "$(DATA_DIR)/raw/NTUH/human_label" \
		--output_csv "$(DATA_DIR)/csvs/pano_ntuh_human_label_{}.csv" \
		--verbosity $(VERBOSITY)

# insdet

convert-coco-to-instance-detection: ROOT_DIR = $(RAW_DIR)
convert-coco-to-instance-detection:
	$(RUN_SCRIPT) \
		--data_dir "$(DATA_DIR)" \
		--dataset_prefix $(DATASET_PREFIX) \
		--coco "$(DATA_DIR)/coco/promaton.json"

# semseg

convert-coco-to-detectron2-semseg.v5: MASK_DIR = masks/segmentation-v5
convert-coco-to-detectron2-semseg.v5: DATASET_NAME = pano_raw
convert-coco-to-detectron2-semseg.v5:
	$(MAKE) convert-coco-to-detectron2-semseg \
		MASK_DIR=$(MASK_DIR) \
		DATASET_NAME=$(DATASET_NAME)

convert-coco-to-detectron2-semseg: ROOT_DIR = $(RAW_DIR)
convert-coco-to-detectron2-semseg: MASK_DIR ?= masks/segmentation-v4
convert-coco-to-detectron2-semseg: check-RAW_DIR
convert-coco-to-detectron2-semseg:
	$(RUN_SCRIPT) \
		--data_dir "$(DATA_DIR)" \
		--dataset_prefix $(DATASET_NAME) \
		--mask_dir "$(MASK_DIR)" \
		--num_workers $(CPUS) \
		--verbosity $(VERBOSITY)

### maskdino targets

install-maskdino:
	@$(PY) ./MaskDINO/maskdino/modeling/pixel_decoder/ops/setup.py build install

train-maskdino: MODEL_NAME = $(NEW_NAME)
train-maskdino: DATASET_NAME = pano_train,pano_eval
train-maskdino: CONFIG_NAME ?= config-maskdino-r50.yaml
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
debug-maskdino: CONFIG_NAME ?= config-maskdino-r50.yaml
debug-maskdino: --check-MAIN
debug-maskdino:
	$(RUN_MAIN_DETECTRON2) \
		OUTPUT_DIR $(MODEL_DIR)
		DATASETS.TEST "('pano_debug',)" \
		TEST.EVAL_PERIOD 10

### detectron2 targets

install-detectron2:
	@rm -rf ./detectron2/{build,**/*.so} && \
		$(PIP) install -e ./detectron2 && \
		ln -sf ../../configs ./detectron2/detectron2/model_zoo/

# mvitv2 targets

train-mvitv2: MODEL_NAME = $(NEW_NAME)
train-mvitv2: DATASET_NAME = pano_train,pano_eval
train-mvitv2: CONFIG_NAME ?= mask_rcnn_mvitv2_t_3x.py
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

train-deeplab: MODEL_NAME = $(NEW_NAME)
train-deeplab: DATASET_NAME = pano_semseg_v5
train-deeplab: CONFIG_NAME ?= deeplab-v3.yaml
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

convert-coco-to-yolo: check-ROOT_DIR check-DATASET_PREFIX
convert-coco-to-yolo:
	$(RUN_SCRIPT) \
		--data_dir $(DATA_DIR) \
		--dataset_prefix $(DATASET_PREFIX) \
		--noforce \
		--num_workers $(CPUS) \
		--verbosity $(VERBOSITY)

convert-yolo-labels-to-detectron2-prediction-pt: --check-COMMON check-PREDICTION
convert-yolo-labels-to-detectron2-prediction-pt: POSTFIX =
convert-yolo-labels-to-detectron2-prediction-pt:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--prediction $(PREDICTION) \
		--num_workers $(CPUS) \
		--verbosity $(VERBOSITY)

# when passing `cfg`, all other arguments will be ignored,
# so we dump the config to a temp file and append the rest
train-yolo: MODEL_NAME ?= $(NEW_NAME)
train-yolo: CONFIG_NAME ?= yolov8m-seg.yaml
train-yolo: TMP_FILE := $(shell mktemp --suffix=.yaml)
train-yolo: check-ROOT_DIR
train-yolo:
	if [ -d $(MODEL_DIR) ]; then \
		cat $(MODEL_DIR)/config.yaml > $(TMP_FILE) && \
			echo "resume: true" >> $(TMP_FILE) ; \
	else \
		mkdir $(MODEL_DIR) && \
			cp $(CONFIG_FILE) $(MODEL_DIR)/config.yaml && \
			cat $(CONFIG_FILE) > $(TMP_FILE) && \
			echo \
				"mode: train" \
				"\ndata: $(DATA_DIR)/yolo/$(DATASET_PREFIX)/metadata.yaml" \
				"\nproject: $(MODEL_DIR_ROOT)" \
				"\nname: ./$(MODEL_NAME)" >> $(TMP_FILE) ; \
	fi && \
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
		save=False \
		save_txt=True \
		save_conf=True

debug-yolo: PYTHON = python -m pdb
debug-yolo: MODEL_DIR_ROOT = /tmp/debug
debug-yolo: MODEL_NAME = $(NEW_NAME)
debug-yolo: CONFIG_NAME = yolov8n-seg.yaml
debug-yolo: check-ROOT_DIR
debug-yolo:
	$(YOLO_TRAIN) \
		cfg="$(CONFIG_FILE)" \
		mode=train \
		data="$(DATA_DIR)/yolo/$(DATASET_PREFIX)/metadata.yaml" \
		project="$(MODEL_DIR_ROOT)" \
		name="./$(MODEL_NAME)" \
		cache=false \
		workers=0

### overall targets

install: install-maskdino install-detectron2
train: check-ARCH train-$(ARCH)
test: check-ARCH test-$(ARCH)
debug: check-ARCH debug-$(ARCH)

coco-annotator:
	cd coco-annotator && \
		docker compose up --build --detach

--check-postprocess: --check-COMMON check-SEMSEG_RESULT_NAME check-SEMSEG_DATASET_NAME

postprocess: --check-postprocess
postprocess:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--prediction $(RAW_PREDICTION) \
		--semseg_result_dir $(RESULT_DIR_ROOT)/$(SEMSEG_RESULT_NAME) \
		--semseg_dataset_name $(SEMSEG_DATASET_NAME) \
		--output_prediction $(PREDICTION) \
		--output_csv $(RESULT_CSV) \
		--min_score $(MIN_SCORE) \
		--min_area 0 \
		--min_iom 0.3 \
		--missing_scoring_method $(MISSING_SCORING_METHOD) \
		--finding_scoring_method $(FINDING_SCORING_METHOD) \
		--nosave_predictions \
		--num_workers $(CPUS)

postprocess.gt-det: --check-postprocess
postprocess.gt-det: RESULT_NAME = $(DATASET_NAME)
postprocess.gt-det: POSTFIX = .postprocessed-with-$(SEMSEG_RESULT_NAME)
postprocess.gt-det:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--use_gt_as_prediction \
		--semseg_result_dir $(RESULT_DIR_ROOT)/$(SEMSEG_RESULT_NAME) \
		--semseg_dataset_name $(SEMSEG_DATASET_NAME) \
		--output_csv $(RESULT_CSV) \
		--min_iom 1.0 \
		--missing_scoring_method SHARE_NOBG \
		--finding_scoring_method SHARE_NOBG_NOMUL_MISSING \
		--nosave_predictions \
		--num_workers $(CPUS)

postprocess.gt-all: --check-postprocess
postprocess.gt-all: RESULT_NAME = $(DATASET_NAME)
postprocess.gt-all: POSTFIX = .postprocessed-with-gt.csv
postprocess.gt-all:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--use_gt_as_prediction \
		--semseg_dataset_name $(SEMSEG_DATASET_NAME) \
		--use_semseg_gt_as_prob \
		--output_csv $(RESULT_CSV) \
		--min_iom 1.0 \
		--missing_scoring_method SHARE_NOBG \
		--finding_scoring_method SHARE_NOBG_NOMUL_MISSING \
		--nosave_predictions \
		--num_workers $(CPUS)

visualize: --check-COMMON
visualize: POSTFIX =
visualize: MIN_SCORE = 0.01
visualize:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--prediction $(PREDICTION) \
		--visualize_dir $(VISUALIZE_DIR) \
		--visualize_subset \
		--min_score $(MIN_SCORE) \
		--noforce \
		--num_workers $(CPUS)

visualize.gt: --check-COMMON
visualize.gt: VISUALIZE_DIR = visualize
visualize.gt: RESULT_NAME = $(DATASET_NAME)
visualize.gt:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--use_gt_as_prediction \
		--visualize_dir $(VISUALIZE_DIR) \
		--visualize_subset \
		--noforce \
		--num_workers $(CPUS)

visualize-semseg: --check-COMMON
visualize-semseg: VISUALIZE_DIR = visualize
visualize-semseg:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--prediction $(SEMSEG_PREDICTION) \
		--visualize_dir $(VISUALIZE_DIR) \
		--noforce \
		--num_workers $(CPUS)

visualize-coco: --check-COMMON --check-COCO
visualize-coco:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--prediction $(PREDICTION) \
		--coco_annotator_url $(COCO_ANNOTATOR_URL)

evaluate-auroc: --check-COMMON
evaluate-auroc:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--csv $(RESULT_CSV) \
		--golden_csv_path "$(DATA_DIR)/csvs/$(DATASET_PREFIX)_golden_label.csv" \
		--evaluation_dir $(EVALUATION_DIR) \
		--title "$(DATASET_TITLE)"

evaluate-auroc.with-human: --check-COMMON
evaluate-auroc.with-human:
	$(RUN_SCRIPT) \
		$(COMMON_ARGS) \
		--csv $(RESULT_CSV) \
		--golden_csv_path "$(DATA_DIR)/csvs/$(DATASET_PREFIX)_golden_label.csv" \
		--human_csv_path "$(DATA_DIR)/csvs/$(DATASET_PREFIX)_human_label_{}.csv" \
		--evaluation_dir $(EVALUATION_DIR).with-human \
		--title "$(DATASET_TITLE)"

compare: IMAGE_HEIGHT ?= 600
compare: HTML_PATH ?= $(RESULT_DIR_ROOT)/$(DATASET_NAME)/visualize.html
compare: check-ROOT_DIR check-IMAGE_PATTERNS
	$(RUN_SCRIPT) \
		--image_patterns $(IMAGE_PATTERNS) \
		--output_html $(HTML_PATH) \
		--height $(IMAGE_HEIGHT) \
		--verbosity $(VERBOSITY)

compile-stats:
	$(RUN_SCRIPT) \
		--data_dir $(DATA_DIR) \
		--dicom_dir $(RAW_DIR)/data/dicoms \
		--dataset_name $(DATASET_NAME) \
		--verbosity $(VERBOSITY)
