'''
This code is used to find the best validation epoch and to calculate the performance of the model.
How to run: 
$ python get_final_performance_numbers.py results/interaction_prediction_reddit.txt 

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

import sys
import numpy as np

fname = sys.argv[1]

validation_performances = []
test_performances = []
val = []
test = []
f = open(fname, "r")
idx = -1
for l in f:
    if "Validation performance of epoch" in l:
        if val != []:
            validation_performances.append(val)
            test_performances.append(test)
        idx = int(l.strip().split("epoch ")[1].split()[0])
        val = [idx]
        test = [idx]
        
    if "Validation:" in l:
        val.append(float(l.strip().split(": ")[-1]))
    if "Test:" in l:
        test.append(float(l.strip().split(": ")[-1]))

if val != []:
    validation_performances.append(val)
    test_performances.append(test)

validation_performances = np.array(validation_performances)
test_performances = np.array(test_performances)

if "interaction" in fname:
    metrics = ['Mean Reciprocal Rank', 'Recall@10']
else:
    metrics = ['AUC']

print('\n\n*** For file: %s ***' % fname)
best_val_idx = np.argmax(validation_performances[:,1])
print("Best validation epoch: %d" % best_val_idx)
print('\n\n*** Best validation performance (epoch %d) ***' % best_val_idx)
for i in xrange(len(metrics)):
    print(metrics[i] + ': ' + str(validation_performances[best_val_idx][i+1]))

print('\n\n*** Final model performance on the test set, i.e., in epoch %d ***' % best_val_idx)
for i in xrange(len(metrics)):
    print(metrics[i] + ': ' + str(test_performances[best_val_idx][i+1]))
