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
  echo "Running Experiment on Cross-dataset Source->UNBC Target->Biovid Subject-$i"
  CUDA_VISIBLE_DEVICES=$1 python methods/src_selection_cross_dataset.py --tar_subject=$i --top_s=$2
  i=$((i + 1))
done

echo "Cross-dataset all target subject adaptation completed."