#! /usr/bin/bash

# for output
mkdir -p /data/PointNet_results

# Python3 command
PY3="nice -n 10 python"

# first, traing a classifier.
# ModelNet categories are given in './sampledata/modelnet40_half1.txt' (as examaple)
${PY3} train_classifier.py \
 -o /data/PointNet_results/ex1_classifier_0915 \
 -i /data/work/pointnet/ModelNet40 \
 -c ./sampledata/modelnet40_half1.txt \
 -l /data/PointNet_results/ex1_classifier_0915.log \
 --device cuda:1

# the one of the results is '${HOME}/results/ex1_classifier_0915_feat_best.pth'
# this file is the model that computes PointNet feature.

# train PointNet-LK. fine-tune the PointNet feature for classification (the above file).
${PY3} train_pointlk.py \
 -o /data/PointNet_results/ex1_pointlk_0915 \
 -i /home/yasuhiro/work/pointnet/ModelNet40 \
 -c ./sampledata/modelnet40_half1.txt \
 -l /data/PointNet_results/ex1_pointlk_0915.log \
 --transfer-from /data/PointNet_results/ex1_classifier_0915_feat_best.pth \
 --epochs 400 \
 --device cuda:1

# the trained model:
# ${HOME}/results/ex1_pointlk_0915_model_best.pth

#EOF
