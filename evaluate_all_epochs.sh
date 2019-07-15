#!/bin/bash

: '
This code evaluates the performance of the model for all epochs.
Then it runs the code that find the best validation epoch and uses it to calculate the performance of the model.

To run the code for interaction prediction on the reddit dataset: 
$ ./evaluate_all_epochs.sh reddit interaction

To run the code for state change prediction on the reddit dataset: 
$ ./evaluate_all_epochs.sh reddit state
'

network=$1
type=$2
interaction="interaction"

idx=0
while [ $idx -le 49 ]
do
    echo $idx
    if [ $type == "$interaction" ]; then
	python2.7 evaluate_interaction_prediction.py --network $network --model jodie --epoch ${idx}
    else
	python2.7 evaluate_state_change_prediction.py --network $network --model jodie --epoch ${idx}
    fi
    (( idx+=1 ))
done 


if [ $type == "$interaction" ]; then
    python get_final_performance_numbers.py results/interaction_prediction_${network}.txt
else
    python get_final_performance_numbers.py results/state_change_prediction_${network}.txt
fi
