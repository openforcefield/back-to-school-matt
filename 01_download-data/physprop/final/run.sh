#!/bin/bash

mkdir -p logs
mkdir -p output
mkdir -p images

python filter-data-training.py -np 8 > logs/filter-data-training.log 2>&1
