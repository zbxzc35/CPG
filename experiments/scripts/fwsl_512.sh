#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

is_next=false
for var in "$@"
do
	if ${is_next}
	then
		EXP_DIR=${var}
		break
	fi
	if [ ${var} == "EXP_DIR" ]
	then
		is_next=true
	fi
done

if $NET ~= "VGG16"
then
	echo "only support VGG16 network"
	exit 0
fi

case $DATASET in
	pascal_voc)
		TRAIN_IMDB="voc_2007_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		ITERS=30
		ITERS2=200
		;;
	pascal_voc10)
		TRAIN_IMDB="voc_2010_trainval"
		TEST_IMDB="voc_2010_test"
		PT_DIR="pascal_voc"
		ITERS=30
		;;
	pascal_voc07+12)
		TRAIN_IMDB="voc_2007+2012_trainval"
		TEST_IMDB="voc_2007_test"
		PT_DIR="pascal_voc"
		ITERS=30
		;;
	coco)
		TRAIN_IMDB="coco_2014_train"
		TEST_IMDB="coco_2014_minival"
		PT_DIR="coco"
		ITERS=280000
		;;
	*)
		echo "No dataset given"
		exit
		;;
esac

mkdir -p "experiments/logs/${EXP_DIR}"
LOG="experiments/logs/${EXP_DIR}/${0##*/}_${NET}_${EXTRA_ARGS_SLUG}_`date +'%Y-%m-%d_%H-%M-%S'`.log"
LOG=`echo "$LOG" | sed 's/\[//g' | sed 's/\]//g'`
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

echo ---------------------------------------------------------------------
git log -1
git submodule foreach 'git log -1'
echo ---------------------------------------------------------------------

#=========================================================================
#第一步
echo ---------------------------------------------------------------------
echo showing the solver file:
cat "models/${PT_DIR}/${NET}/cpg/solver.prototxt"
echo ---------------------------------------------------------------------
time ./tools/wsl/train_net.py --gpu ${GPU_ID} \
	--solver models/${PT_DIR}/${NET}/cpg/solver.prototxt \
	--weights data/imagenet_models/${NET}.v2.caffemodel \
	--imdb ${TRAIN_IMDB} \
	--iters ${ITERS} \
	--cfg experiments/cfgs/cpg.yml \
	${EXTRA_ARGS} \
	EXP_DIR ${EXP_DIR}/CPG \
	TRAIN.SCALES [512] \
	TEST.SCALES [512]

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} |tail -n 2 | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/wsl/test_net.py --gpu ${GPU_ID} \
	--def models/${PT_DIR}/${NET}/cpg/test.prototxt \
	--net ${NET_FINAL} \
	--imdb ${TEST_IMDB} \
	--cfg experiments/cfgs/cpg.yml \
	${EXTRA_ARGS} \
	EXP_DIR ${EXP_DIR}/CPG \
	TRAIN.SCALES [512] \
	TEST.SCALES [512]

time ./tools/wsl/test_net.py --gpu ${GPU_ID} \
	--def models/${PT_DIR}/${NET}/cpg/test.prototxt \
	--net ${NET_FINAL} \
	--imdb ${TRAIN_IMDB} \
	--cfg experiments/cfgs/cpg.yml \
	${EXTRA_ARGS} \
	EXP_DIR ${EXP_DIR}/CPG \
	TRAIN.SCALES [512] \
	TEST.SCALES [512]

#=========================================================================
#第二步
python ./tools/fwsl/ssd_pascalvoc07_512.py ${EXP_DIR}/SSD

echo ---------------------------------------------------------------------
echo showing the solver file:
cat "output/${EXP_DIR}/SSD/solver.prototxt"
echo ---------------------------------------------------------------------
time ./tools/ssd/train_net.py --gpu ${GPU_ID} \
	--solver output/${EXP_DIR}/SSD/solver.prototxt \
	--weights data/imagenet_models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel \
	--imdb ${TRAIN_IMDB} \
	--iters ${ITERS} \
	--cfg experiments/cfgs/ssd.yml \
	${EXTRA_ARGS} \
	TRAIN.PROPOSAL_METHOD pseudo_gt \
	TRAIN.PSEUDO_PATH output/${EXP_DIR}/CPG/${TRAIN_IMDB}/VGG16_iter_30/detections_o.pkl

#=========================================================================
#第三步
python ./tools/fwsl/fwsl_pascalvoc07_512.py ${EXP_DIR}/FWSL

./tools/fwsl/fc6fc7_to_wsl.py \
	models/${PT_DIR}/${NET}/cpg/test.prototxt \
	output/${EXP_DIR}/CPG/${TRAIN_IMDB}/VGG16_iter_30.caffemodel \
	models/${PT_DIR}/${NET}/cpg/predict_module.prototxt \
	output/${EXP_DIR}/CPG/${TRAIN_IMDB}/VGG16_iter_30_predict.caffemodel

echo ---------------------------------------------------------------------
echo showing the solver file:
cat "output/${EXP_DIR}/FWSL/solver.prototxt"
echo ---------------------------------------------------------------------
time ./tools/fwsl/train_net.py --gpu ${GPU_ID} \
	--solver output/${EXP_DIR}/FWSL/solver.prototxt \
	--weights output/${EXP_DIR}/CPG/${TRAIN_IMDB}/VGG16_iter_30_predict.caffemodel,output/${EXP_DIR}/SSD/VGG_VOC2007_iter_80000.caffemodel \
	--imdb ${TRAIN_IMDB} \
	--iters ${ITERS} \
	--cfg experiments/cfgs/fwsl.yml \
	${EXTRA_ARGS} \
	EXP_DIR ${EXP_DIR}/FWSL \
	TRAIN.SCALES [512] \
	TEST.SCALES [512]
