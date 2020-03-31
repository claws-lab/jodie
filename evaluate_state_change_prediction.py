'''
This code evaluates the validation and test performance in an epoch of the model trained in jodie.py.
The task is: user state change prediction, i.e.,  when the state of a user changes from one to another, say normal to abnormal.

To calculate the performance for one epoch:
$ python evaluate_state_change_prediction.py --network reddit --model jodie --epoch 49

To calculate the performance for all epochs, use the bash file, evaluate_all_epochs.sh, which calls this file once for every epoch.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from library_data import *
from library_models import *
from sklearn.metrics import roc_auc_score

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Network name')
parser.add_argument('--model', default='jodie', help="Model name")
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epoch', default=50, type=int, help='Epoch id to load')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users in addition to the next interaction prediction. False otherwise. By default, set to True. MUST BE THE SAME AS THE ONE USED IN TRAINING.') 
args = parser.parse_args()
args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')
if args.network == "lastfm":
    print("No state change prediction for %s" % args.network)
    sys.exit(0)
    
# SET GPU
args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# CHECK IF THE OUTPUT OF THE EPOCH IS ALREADY PROCESSED. IF SO, MOVE ON.
output_fname = "results/state_change_prediction_%s.txt" % args.network
if os.path.exists(output_fname):
    f = open(output_fname, "r")
    search_string = 'Test performance of epoch %d' % args.epoch
    for l in f:
        l = l.strip()
        if search_string in l:
            print("Output file already has results of epoch %d" % args.epoch)
            sys.exit(0)
    f.close()

# LOAD NETWORK
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, \
 item2id, item_sequence_id, item_timediffs_sequence, \
 timestamp_sequence, \
 feature_sequence, \
 y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_features = len(feature_sequence[0])
num_users = len(user2id)
num_items = len(item2id) + 1
true_labels_ratio = len(y_true)/(sum(y_true)+1)
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))
        
# SET TRAIN, VALIDATION, AND TEST BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion + 0.1))
test_end_idx = int(num_interactions * (args.train_proportion + 0.2))

# SET BATCHING TIMESPAN
'''
Timespan indicates how frequently the model is run and updated. 
All interactions in one timespan are processed simultaneously. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
At the end of each timespan, the model is updated as well. So, longer timespan means less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500 

# INITIALIZE MODEL PARAMETERS
model = JODIE(args, num_features, num_users, num_items).cuda()
weight = torch.Tensor([1,true_labels_ratio]).cuda()
crossEntropyLoss = nn.CrossEntropyLoss(weight=weight)
MSELoss = nn.MSELoss()

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# LOAD THE MODEL
model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, args.epoch)
if train_end_idx != train_end_idx_training:
    sys.exit('Training proportion during training and testing are different. Aborting.')

# SET THE USER AND ITEM EMBEDDINGS TO THEIR STATE AT THE END OF THE TRAINING PERIOD
set_embeddings_training_end(user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, user_sequence_id, item_sequence_id, train_end_idx) 

# LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
item_embeddings = item_embeddings.clone()
item_embeddings_static = item_embeddings_dystat[:, args.embedding_dim:]
item_embeddings_static = item_embeddings_static.clone()

user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
user_embeddings = user_embeddings.clone()
user_embeddings_static = user_embeddings_dystat[:, args.embedding_dim:]
user_embeddings_static = user_embeddings_static.clone()

validation_predicted_y = []
test_predicted_y = []
validation_true_y = []
test_true_y = []

''' 
Here we use the trained model to make predictions for the validation and testing interactions.
The model does a forward pass from the start of validation till the end of testing.
For each interaction, the trained model is used to predict the embedding of the item it will interact with. 
This is used to calculate the rank of the true item the user actually interacts with.

After this prediction, the errors in the prediction are used to calculate the loss and update the model parameters. 
This simulates the real-time feedback about the predictions that the model gets when deployed in-the-wild. 
Please note that since each interaction in validation and test is only seen once during the forward pass, there is no data leakage. 
'''
tbatch_start_time = None
loss = 0
# FORWARD PASS
print("*** Making state change predictions by forward pass (no t-batching) ***")
with trange(train_end_idx, test_end_idx) as progress_bar:
    for j in progress_bar:
        progress_bar.set_description('%dth interaction for validation and testing' % j)

        # LOAD INTERACTION J
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]
        feature = feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        timestamp = timestamp_sequence[j]
        if not tbatch_start_time:
            tbatch_start_time = timestamp
        itemid_previous = user_previous_itemid_sequence[j]

        # LOAD USER AND ITEM EMBEDDING
        user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
        user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
        item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]
        item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]
        feature_tensor = Variable(torch.Tensor(feature).cuda()).unsqueeze(0)
        user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
        item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).cuda()).unsqueeze(0)
        item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]

        # PROJECT USER EMBEDDING
        user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
        user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.cuda.LongTensor([itemid_previous])], user_embedding_static_input], dim=1)
        
        # PREDICT ITEM EMBEDDING
        predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

        # CALCULATE PREDICTION LOSS
        loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_input], dim=1).detach())

        # UPDATE USER AND ITEM EMBEDDING
        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update') 
        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update') 

        # SAVE EMBEDDINGS
        item_embeddings[itemid,:] = item_embedding_output.squeeze(0) 
        user_embeddings[userid,:] = user_embedding_output.squeeze(0) 
        user_embeddings_timeseries[j, :] = user_embedding_output.squeeze(0)
        item_embeddings_timeseries[j, :] = item_embedding_output.squeeze(0)

        # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
        loss += MSELoss(item_embedding_output, item_embedding_input.detach())
        loss += MSELoss(user_embedding_output, user_embedding_input.detach())

        # CALCULATE STATE CHANGE LOSS
        if args.state_change:
            loss += calculate_state_prediction_loss(model, [j], user_embeddings_timeseries, y_true, crossEntropyLoss) 

        # UPDATE THE MODEL IN REAL-TIME USING ERRORS MADE IN THE PAST PREDICTION
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # RESET LOSS FOR NEXT T-BATCH
            loss = 0
            item_embeddings.detach_()
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_() 
            user_embeddings_timeseries.detach_() 

        # PREDICT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
        prob = model.predict_label(user_embedding_output)

        # ADD PREDICTION TO VALIDATION OR TEST ARRAYS
        if j < test_start_idx:
            validation_predicted_y.extend(prob.data.cpu().numpy())
            validation_true_y.extend([y_true[j]])
        else:
            test_predicted_y.extend(prob.data.cpu().numpy())
            test_true_y.extend([y_true[j]])

# CALCULATE THE PERFORMANCE METRICS
validation_predicted_y = np.array(validation_predicted_y)
test_predicted_y = np.array(test_predicted_y)

performance_dict = dict()
auc = roc_auc_score(validation_true_y, validation_predicted_y[:,1])
performance_dict['validation'] = [auc]

auc = roc_auc_score(test_true_y, test_predicted_y[:,1])
performance_dict['test'] = [auc]

# PRINT AND SAVE THE PERFORMANCE METRICS
fw = open(output_fname, "a")
metrics = ['AUC']

print('\n\n*** Validation performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Validation performance of epoch %d ***\n' % args.epoch)

for i in xrange(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['validation'][i]))
    fw.write("Validation: " + metrics[i] + ': ' + str(performance_dict['validation'][i]) + "\n")

print('\n\n*** Test performance of epoch %d ***' % args.epoch)
fw.write('\n\n*** Test performance of epoch %d ***\n' % args.epoch)
for i in xrange(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['test'][i]))
    fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")

fw.flush()
fw.close()

