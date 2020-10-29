#!/bin/bash

# generate perturbations for each object (for each 'ModelNet40/[category]/test/*')

# for output
OUTDIR=/data/PointNet_results/gt
mkdir -p ${OUTDIR}

# Python3 command
PY3="nice -n 10 python"

# categories for testing
CMN="-i /data/ModelNet40/ -c ./sampledata/modelnet40_half2.txt"


${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_010.csv --mag 1.0


#EOF
