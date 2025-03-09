#!/bin/bash

conda activate TTC

cd ./

python code/test_time_counterattack.py \
    --batch_size 256 \
    --test_attack_type 'pgd' \
    --test_eps 1 \
    --test_numsteps 10 \
    --test_stepsize 1 \
    --outdir 'TTC_results' \
    --seed 1 \
    --ttc_eps 4 \
    --beta 2 \
    --tau_thres 0.2 \
    --ttc_numsteps 2
