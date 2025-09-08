#!/bin/bash

mkdir -p logs

python write-options.py -n 1000 > logs/write-options.log 2>&1
