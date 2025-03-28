#!/bin/sh

Check if an argument is provided
if [ -z "$2" ]; then
  echo "Usage: $0 <GPU_ID>, $1 <Top_s>"
  exit 1
fi

# Number of target subject
N=4

# Loop from 1 to N
i=0
while [ $i -le $N ]
do
  echo "Running Experiment on UNBC Target Subject-$i"
  CUDA_VISIBLE_DEVICES=$1 python methods/self_paced_sub_specific_msda.py --tar_subject=$i --top_s=$2
  i=$((i + 1))
done

echo "UNBC-McMaster all target subject adaptation completed."