'''
This code is used to find the best validation epoch and use it to calculate the performance of the model.
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
print "All validation performance numbers:", validation_performances
print "All test performance numbers:", test_performances

print "\n\n"
print "For file: %s" % fname
best_val_idx = np.argmax(validation_performances[:,1])
print "Best validation performance:", validation_performances[best_val_idx]
print "Corresponding test performance:", test_performances[best_val_idx]
