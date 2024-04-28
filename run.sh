#!/bin/bash

export RAW_DIR=/mnt/md0/data/PANO
export ROOT_DIR=/mnt/hdd/PANO
export CPUS=1
export CUDA_VISIBLE_DEVICES=0

export DEBUG=
# export DEBUG=pdb
# export DEBUG=memray
# export DEBUG=cProfile

### utility functions

function _NEW_NAME() {
  date "+%Y-%m-%d-%H%M%S"
}

function _CHECK_VARIABLE() {
  if [[ -z "${!1}" ]] ; then
    echo "$1 is required"
    exit 1
  fi
}

function _EXIT() {
  clear
  exit $1
}

function _CHECK_EXIT_CODE() {
  OK=0
  CANCEL=1
  ESCAPE=255

  EXIT_CODE=$1
  case ${EXIT_CODE} in
    ${OK} )
      ;;

    ${CANCEL} | ${ESCAPE} )
      _EXIT ${EXIT_CODE}
      ;;

    * )
      echo "Unknown exit code: ${EXIT_CODE}"
      _EXIT ${EXIT_CODE}
      ;;

  esac
}

function __DEFAULT_SET_NAME() {
  echo "Please invoke _IMPORT_EXP_SH to introduce $1 function"
  _EXIT 1
}

function _SET_DATASET_NAME() { __DEFAULT_SET_NAME "_SET_DATASET_NAME"; }
function _SET_MODEL_NAME() { __DEFAULT_SET_NAME "_SET_MODEL_NAME"; }
function _SET_RESULT_NAME() { __DEFAULT_SET_NAME "_SET_RESULT_NAME"; }

function _IMPORT_EXP_SH() {
  # See: https://docs.google.com/spreadsheets/d/1EwkoX-EZM-vcP1r3a7J2IknsEmleFiucXPwAa5YvKpY

  EXP_SH=.experiment.sh
  URL=${1:-"https://docs.google.com/spreadsheets/d/e/2PACX-1vSD6m_E8CQMX4Fm850d3-RZ1fg7gHEOZmouc5I3rIkzObDUk2sWhHqcTusVeattKRY1HjxAImwvjo2i/pub?gid=1641585683&single=true&output=tsv"}

  if [[ -z "${USE_CACHED_EXP_SH}" ]] || [[ ! -f "${EXP_SH}" ]] ; then
    echo "Downloading experiment configuration from Google Sheets"
    curl -L -s ${URL} | sed 's/\r//g' > ${EXP_SH}
  fi
  . ${EXP_SH}
}

### dialog functions

function _SELECT_VARIABLE() {
  VAR_NAME=$1
  AVAILABLE_VALUES=(${@:2})

  ITEMS=()
  for VALUE in ${AVAILABLE_VALUES[@]} ; do
    ITEMS+=("${VALUE}" "${VALUE}")
  done

  VAR_VALUE=$( \
    dialog --clear \
      --menu "Select value for ${VAR_NAME}" 0 0 0 "${ITEMS[@]}" \
      3>&2 2>&1 1>&3 \
  )
  _CHECK_EXIT_CODE $?

  eval export ${VAR_NAME}="${VAR_VALUE}"
}

function _INPUT_VARIABLE() {
  VAR_NAME=$1
  _CHECK_VARIABLE "VAR_NAME"

  VAR_VALUE=$( \
    dialog --clear \
      --inputbox "Enter ${VAR_NAME}" 0 0 \
      3>&2 2>&1 1>&3 \
  )
  _CHECK_EXIT_CODE $?

  eval export ${VAR_NAME}="${VAR_VALUE}"
}

function __INPUT_VARIABLES() {
  TITLE=$1
  NAME_WIDTH=$2
  VALUE_WIDTH=$3
  VAR_NAMES=(${@:4})

  ITEMS=()
  for (( i=1; i<=${#VAR_NAMES[@]}; i++ )) ; do
    VAR_NAME=${VAR_NAMES[$i-1]}
    VAR_VALUE=${!VAR_NAME:-""}

    ITEMS+=("${VAR_NAME}" ${i} 1 "${VAR_VALUE}" ${i} ${NAME_WIDTH} ${VALUE_WIDTH} 0)
  done

  exec 3>&1
  VAR_VALUES=$( \
    dialog --clear --output-separator "," \
      --form "${TITLE}" 0 0 0 "${ITEMS[@]}" \
      2>&1 1>&3 \
  )
  _CHECK_EXIT_CODE $?
  exec 3>&-

  for (( i=1; i<=${#VAR_NAMES[@]}; i++ )) ; do
    VAR_NAME=${VAR_NAMES[$i-1]}
    VAR_VALUE=$(echo ${VAR_VALUES} | cut -d "," -f ${i})

    eval export ${VAR_NAME}="${VAR_VALUE}"
  done
}

function _INPUT_VARIABLES() {
  __INPUT_VARIABLES "Input variables" 24 48 $@
}

function _CONFIRM_VARIABLES() {
  __INPUT_VARIABLES "Confirm variables" 24 -48 $@
}

### main logic

function _TARGET_CREATE_MODEL() {
  MODEL_NAME=$(_NEW_NAME)
  ARCH=yolo
  DATASET_PREFIX=pano
  CONFIG_NAME=insdet-v2-yolov8.yaml

  while : ; do
    _INPUT_VARIABLES \
      "MODEL_NAME" \
      "ARCH" \
      "DATASET_PREFIX" \
      "CONFIG_NAME"

    # perform checks against the variables

    break
  done
}

function _TARGET_CREATE_EVALUATION() {
  _IMPORT_EXP_SH
  _INPUT_VARIABLES \
    "RESULT_NAME" \
    "SEMSEG_RESULT_NAME"

  if [[ ! -z "${DATASET_NAME}" ]] ; then
    FORCE_DATASET_NAME=${DATASET_NAME}
  fi

  INSDET_RESULT_NAME=${RESULT_NAME}
  _SET_RESULT_NAME ${SEMSEG_RESULT_NAME}
  SEMSEG_DATASET_NAME=${DATASET_NAME}
  _SET_RESULT_NAME ${INSDET_RESULT_NAME}

  if [[ ! -z "${FORCE_DATASET_NAME}" ]] ; then
    echo "DATASET_NAME specified, please ensure ${FORCE_DATASET_NAME} is a subset of ${DATASET_NAME}"

    DATASET_NAME=${FORCE_DATASET_NAME}
  fi

  while : ; do
    _INPUT_VARIABLES \
      "RESULT_NAME" \
      "SEMSEG_RESULT_NAME" \
      "DATASET_NAME" \
      "SEMSEG_DATASET_NAME" \
      "CPUS" \
      "DEBUG"

    # perform checks against the variables

    break
  done

  _SET_MODEL_NAME ${MODEL_NAME}
  _SET_DATASET_NAME ${DATASET_NAME}

  export SEMSEG_DATASET_NAME
}

function _MAIN() {

  if [[ -z "${TARGET}" ]] ; then
    _SELECT_VARIABLE "TARGET" \
      "train" \
      "test" \
      "visualize" \
      "visualize-semseg" \
      "visualize-coco" \
      "postprocess" \
      "evaluate-auroc" \
      "evaluate-auroc.with-human" \
      "compile-stats" \
      "convert-coco-to-instance-detection" \
      "compare" \
      "plot-performances" \
      "plot-agreements"
  fi

  case "${TARGET}" in

    "train" ) _TARGET_CREATE_MODEL ;;

    "test" )

      _IMPORT_EXP_SH
      _INPUT_VARIABLE "RESULT_NAME"

      if [[ -z "${RESULT_NAME}" ]] ; then
        RESULT_NAME=$(_NEW_NAME)
        MODEL_NAME=2024-04-12-075958
        DATASET_NAME=pano_semseg_v5_ntuh_test
        MODEL_CHECKPOINT=model_0099999.pth
        MAX_OBJS=300
		    MIN_SCORE=0.0001
		    MIN_IOU=0.5

		  else
		    _SET_RESULT_NAME ${RESULT_NAME}
        _SET_MODEL_NAME ${MODEL_NAME}
        _SET_DATASET_NAME ${DATASET_NAME}

      fi

      _INPUT_VARIABLES \
        "RESULT_NAME" \
        "MODEL_NAME" \
        "MODEL_CHECKPOINT" \
        "DATASET_NAME" \
        "MAX_OBJS" \
        "MIN_SCORE" \
        "MIN_IOU"

      _SET_MODEL_NAME ${MODEL_NAME}
      _SET_DATASET_NAME ${DATASET_NAME}

      if [[ "${ARCH}" == "yolo" ]] ; then
        ADDITIONAL_TARGETS+=("convert-yolo-labels-to-detectron2-prediction-pt")

      fi

      ;;

    "visualize" \
    | "visualize-semseg" \
    | "visualize-coco" \
    )

      _IMPORT_EXP_SH
      _INPUT_VARIABLE "RESULT_NAME"

      _SET_RESULT_NAME ${RESULT_NAME}
      _SET_MODEL_NAME ${MODEL_NAME}

      _INPUT_VARIABLES \
        "RESULT_NAME" \
        "DATASET_NAME" \
        "MODEL_NAME" \
        "CPUS"

      ;;

    "postprocess" \
    | "evaluate-auroc" \
    | "evaluate-auroc.with-human" \
    )

      _TARGET_CREATE_EVALUATION

      ;;

    "compile-stats" )

      _SELECT_VARIABLE "DATASET_NAME" \
        "pano_eval_v2" \
        "pano_test_v2_1" \
        "pano_ntuh_test_v2"

      ;;

    "convert-coco-to-instance-detection" )

      _INPUT_VARIABLE "DATASET_PREFIX"

      ;;

    "compare" )

      _SELECT_VARIABLE "DATASET_NAME" \
        "pano_train" \
        "pano_eval_v2" \
        "pano_test_v2_1" \
        "pano_ntuh_test_v2"

      case "${DATASET_NAME}" in

        "pano_train" )

		      RAW_IMAGE_PATTERNS="
			      ${ROOT_DIR}/data/images/PROMATON/*.jpg,
			      ${ROOT_DIR}/results/${DATASET_PREFIX}/visualize/TRP10*.tooth.jpg,
			      ${ROOT_DIR}/results/${DATASET_PREFIX}/visualize/*.findings.jpg,
			      ${ROOT_DIR}/data/masks/segmentation-v5/PROMATON/*_vis.png,
			    "

			    ;;

			  "pano_eval_v2" )

		      RAW_IMAGE_PATTERNS="
			      ${ROOT_DIR}/data/images/PROMATON/*.jpg,
			      ${ROOT_DIR}/results/${DATASET_PREFIX}/visualize/*.tooth.jpg,
			      ${ROOT_DIR}/results/2024-04-15-043941/visualize/*.tooth.jpg,
			      ${ROOT_DIR}/results/${DATASET_PREFIX}/visualize/*.findings.jpg,
			      ${ROOT_DIR}/results/2024-04-15-043941/visualize/*.findings.jpg,
			    "

			    ;;

			  "pano_test_v2_1" )

			    RAW_IMAGE_PATTERNS="
			      ${ROOT_DIR}/data/images/PROMATON/*.jpg,
			      ${ROOT_DIR}/results/2024-04-15-042952/visualize/*.tooth.jpg,
			      ${ROOT_DIR}/results/2024-04-15-042952/visualize/*.findings.jpg,
			    "

			    ;;

			  "pano_ntuh_test_v2" )

          RAW_IMAGE_PATTERNS="
			      ${ROOT_DIR}/data/images/NTUH/*.jpg,
			      ${ROOT_DIR}/results/2024-02-20-050603/visualize/*.tooth.jpg,
			      ${ROOT_DIR}/results/2024-04-19-030353/visualize/*.tooth.jpg,
			      ${ROOT_DIR}/results/2024-04-13-022434/visualize/*.heatmap.jpg,
			      ${ROOT_DIR}/results/2024-02-20-050603/visualize/*.findings.jpg,
			      ${ROOT_DIR}/results/2024-04-19-030353/visualize/*.findings.jpg,
          "

          ;;

        * )

          echo "Invalid dataset: ${DATASET_NAME}"
          _EXIT 1

          ;;

      esac

      export IMAGE_PATTERNS="$(echo ${RAW_IMAGE_PATTERNS} | sed 's/ //g' | sed 's/,$//g')"

      ;;

    "plot-performances" )

      RAW_CSVS=(
        "pano_eval_v2:${ROOT_DIR}/results/2024-02-05-154450/evaluation.pano_eval_v2.postprocessed-with-2024-04-14-122002.missing-scoring-SHARE_NOBG.finding-scoring-SCORE_MUL_SHARE_NOBG_NOMUL_MISSING/metrics.csv" \
        "pano_test_v2_1:${ROOT_DIR}/results/2024-03-05-151736/evaluation.pano_test_v2_1.postprocessed-with-2024-04-14-013301.missing-scoring-SHARE_NOBG.finding-scoring-SCORE_MUL_SHARE_NOBG_NOMUL_MISSING/metrics.csv" \
        "pano_ntuh_test_v2:${ROOT_DIR}/results/2024-02-20-050603/evaluation.pano_ntuh_test_v2.postprocessed-with-2024-04-13-022434.missing-scoring-SHARE_NOBG.finding-scoring-SCORE_MUL_SHARE_NOBG_NOMUL_MISSING/metrics.csv"
      )
      export CSVS=${RAW_CSVS[@]}

      ;;

    "plot-agreements" )

      export CSV="results/2024-02-20-050603/evaluation.pano_ntuh_test_v2_1.postprocessed-with-2024-04-13-022434.missing-scoring-SHARE_NOBG.finding-scoring-SCORE_MUL_SHARE_NOBG_NOMUL_MISSING.with-human/evaluation.csv"

      ;;

  esac

  make ${TARGET} ${ADDITIONAL_TARGETS[@]}
}

_MAIN
