'''
This code creates T-batches from a temporal binary network. 
Note that the jodie.py code has the same code in the function to create the t-batches during training, so it does not call this code.

How to run: 
$ python tbatch.py --network reddit 

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from library_data import *
import library_models as lib
from library_models import *

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of data (from beginning) in training')
args = parser.parse_args()
args.datapath = "data/%s.csv" % args.network 

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1 # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error. 
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# OUTPUT FILE FOR THE BATCHES
output_fname = "results/batches_%s.txt" % args.network
fout = open(output_fname, "w")
fout.write("tbatch_id,user_id,item_id,timestamp,state_label,comma_separated_list_of_features\n")

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the JODIE model is trained. 
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm). 
The batches are then used to train JODIE. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500 

# CREATE THE TBATCHES
print("*** Creating T-batches from %d interactions ***" % train_end_idx)
# INITIALIZE TBATCH PARAMETERS
tbatch_start_time = None
tbatch_to_insert = -1
tbatch_full = False

reinitialize_tbatches()
tbatchID = 0
tbatch_interaction_count = 0

total_interactions_count = 0
total_tbatches_count = 0

# CREATE TBATCHES FOR ALL INTERACTIONS IN THE NETWORK
for j in range(num_interactions):
    # READ INTERACTION J
    userid = user_sequence_id[j]
    itemid = item_sequence_id[j]
    timestamp = timestamp_sequence[j]
    feature = feature_sequence[j]
    label = y_true[j]
    user_timediff = user_timediffs_sequence[j]
    item_timediff = item_timediffs_sequence[j]

    # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
    tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1 
    lib.tbatchid_user[userid] = tbatch_to_insert 
    lib.tbatchid_item[itemid] = tbatch_to_insert

    lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
    lib.current_tbatches_user[tbatch_to_insert].append(userid)
    lib.current_tbatches_item[tbatch_to_insert].append(itemid)
    lib.current_tbatches_timestamp[tbatch_to_insert].append(timestamp)
    lib.current_tbatches_feature[tbatch_to_insert].append(feature)
    lib.current_tbatches_label[tbatch_to_insert].append(label)
    lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
    lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
    lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])
    tbatch_interaction_count += 1

    if tbatch_start_time is None:
        tbatch_start_time = timestamp

    # AFTER PROCESSING ALL INTERACTIONS IN A TIMESPAN
    if timestamp - tbatch_start_time > tbatch_timespan or j == num_interactions - 1:
        # AFTER ALL INTERACTIONS IN THE TIME WINDOW ARE CONVERTED TO T-BATCHES, SAVE THEM TO FILE.
        print('Read till interaction %d. This timespan had %d interactions and created %d T-batches.' % (j, tbatch_interaction_count, len(lib.current_tbatches_user)))
        total_tbatches_count += len(lib.current_tbatches_user)
        total_interactions_count += tbatch_interaction_count 

        tbatch_start_time = timestamp # RESET START TIME FOR NEXT TBATCHES
        tbatch_interaction_count = 0

        # ITERATE OVER ALL T-BATCHES
        for tidx in range(len(lib.current_tbatches_user)):
            #print '%d interactions in %d-th T-batch' % (len(lib.current_tbatches_interactionids[tidx]), tidx)
            tbatchID += 1

            # LOAD THE CURRENT TBATCH
            tbatch_interactionids = lib.current_tbatches_interactionids[tidx]
            tbatch_userids = lib.current_tbatches_user[tidx] # "lib.current_tbatches_user[tidx]" has unique elements
            tbatch_itemids = lib.current_tbatches_item[tidx] # "lib.current_tbatches_item[tidx]" has unique elements
            tbatch_timestamps = lib.current_tbatches_timestamp[tidx] # "lib.current_tbatches_item[tidx]" has unique elements
            tbatch_features = lib.current_tbatches_feature[tidx] # "lib.current_tbatches_feature[tidx]" is list of list, so "feature_tensor" is a 2-d tensor
            tbatch_labels = lib.current_tbatches_label[tidx] 

            batch = zip(tbatch_userids, tbatch_itemids, tbatch_timestamps, tbatch_labels, tbatch_features)
            for uid, iid, ts, lbl, feature in batch:
                arr = map(str, [tbatchID, uid, iid, ts, lbl] + feature)
                fout.write(",".join(arr) + "\n")

        reinitialize_tbatches()
        tbatch_to_insert = -1

fout.close()
print("=======================")
print("T-batching complete. Output file: %s." % output_fname)
print("%d interactions were processed, which created %d t-batches." % (total_interactions_count, total_tbatches_count))
print("This is a %.3f%% compression." % ((total_interactions_count - total_tbatches_count)*100.0/total_interactions_count))
print("=======================")
