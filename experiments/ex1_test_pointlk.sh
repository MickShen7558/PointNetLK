#! /usr/bin/bash

# for output
OUTDIR=/data/PointNet_results/ex1/plk
mkdir -p ${OUTDIR}

# Python3 command
PY3="nice -n 10 python"

# categories for testing and the trained model
MODEL=/data/PointNet_results/ex1_pointlk_0915_model_best.pth
CMN="-i /data/ModelNet40/ -c ./sampledata/modelnet40_half2.txt --format wt --pretrained ${MODEL}"

# perturbations
PERDIR=/data/PointNet_results/gt

# test PointNet-LK with given perturbations (see. 'ex1_genrot.sh' for perturbations)
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_010.csv -p ${PERDIR}/pert_010.csv -l ${OUTDIR}/log_010.log


#EOF
