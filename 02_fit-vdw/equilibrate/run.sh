#!/bin/bash

mkdir -p logs

python write-options.py -n 3 > logs/write-options.log 2>&1
