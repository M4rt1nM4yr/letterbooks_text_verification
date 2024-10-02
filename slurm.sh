#!/bin/bash
DEVICES=${3:-1}

echo $1 $2 $DEVICES

sbatch --gres=gpu:a40:$DEVICES --ntasks-per-node=$DEVICES --nodes=1 run.sh $1 "$2" $DEVICES