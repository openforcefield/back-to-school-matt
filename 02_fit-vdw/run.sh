#!/bin/bash

mkdir -p forcefield
mkdir -p output
mkdir -p logs

python generate-forcefield.py -n 3 > logs/generate-forcefield.log 2>&1
