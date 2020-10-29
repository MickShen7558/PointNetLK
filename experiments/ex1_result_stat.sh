#! /usr/bin/bash

# Python3 command
PY3="nice -n 10 python"

# for output
#RES="${HOME}/results/ex1/icp"
RES="${HOME}/PointNet_results/ex1/plk"

# gather 'result_*.csv' to 'result.csv'
${PY3} result_stat.py --hdr > ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_010.csv --val 10 >> ${RES}/result.csv

#EOF