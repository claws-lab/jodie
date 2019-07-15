## JODIE: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks

This repository has the code for the paper:
*Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks*. Srijan Kumar, Xikun Zhang, Jure Leskovec. The paper is published at ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019.

#### Authors: [Srijan Kumar](http://cs.stanford.edu/~srijan) (srijan@cs.stanford.edu), [Xikun Zhang]() (xikunz2@illinois.edu)
#### [Project website with links to the datasets](http://snap.stanford.edu/jodie/)
#### [Link to the paper](https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf)

### Introduction
JODIE is a representation learning framework for temporal interaction networks. Given a sequence of entity-entity interactions, JODIE learns a dynamic embedding trajectory for every entity, which can then be used for various downstream machine learning tasks. JODIE is fast and makes accurate predictions on temporal interaction network.

JODIE can be used for two broad category of tasks:
1. **Interaction prediction**: Which two entities will interact next? This has applications in recommender system and modeling network evolution.
2. **State change prediction**: When does the state of an entity change (e.g., from normal to abnormal)? This has applications in anomaly detection, ban prediction, dropout and churn prediction, fraud and account compromise, and more. 

If you make use of this code, the JODIE algorithm, the T-batch algorithm, or the datasets in your work, please cite the following paper:
```
   @inproceedings{kumar2019predicting,
	 title={Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks},
	 author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},
	 booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},
	 year={2019},
	 organization={ACM}
	}
```

### Motivation 
Temporal interaction networks provide an expressive language to represent time-evolving and dynamic interactions between entities. Representation learning provides a powerful tool to model and reason on networks. However, as networks evolve over time, a single (static) embedding becomes insufficient to represent the changing behavior of the entities and the dynamics of the network.

JODIE is a representation learning framework that embeds every entity in a Euclidean space and their evolution is modeled by an embedding trajectory in this space. JODIE learns to project/forecast the embedding trajectories into the future to make predictions about the entities and their interactions. These trajectories can be trained for downstream tasks, such as recommendations and predictions. JODIE is scalable to large networks by employing a novel t-Batch algorithm that creates batches of independent edges that can be processed simulaneously.

### Setup

To initialize the directories needed to store data and outputs, use the following command. This will create `data/`, `saved_models/`, and `results/` directories.
```
    $ chmod +x initialize.sh
    $ ./initialize.sh
```

To download the datasets used in the paper, use the following command. This will download four datasets under the `data/` directory: `reddit.csv`, `wikipedia.csv`, `mooc.csv`, and `lastfm.csv`.
```
    $ chmod +x download_data.sh
    $ ./download_data.sh
```

### Run the JODIE code

To train the JODIE model, use the following command. This will save a model for every epoch in the `saved_models/<network>/` directory.
```
   $ python jodie.py --network reddit --model jodie --epochs 50
```

This code can be given the following command-line arguments:
1. `--network`: this is the name of the file which has the data in the `data/` directory. The file should be named `<network>.csv`, where `<network> = reddit` in the example above. The dataset format is explained below. This is a required argument. 
2. `--model`: this is the name of the model and the file where the model will be saved in the `saved_models/` directory. Default value: jodie.
3. `--gpu`: this is the id of the gpu where the model is run. Default value: -1 (to run on the GPU with the most free memory).
4. `--epochs`: this is the maximum number of interactions to train the model. Default value: 50.
5. `--embedding_dim`: this is the number of dimensions of the dynamic embedding. Default value: 128.
6. `--train_proportion`: this is the fraction of interactions (from the beginning) that are used for training. The next 10% are used for validation and the next 10% for testing. Default value: 0.8
7. `--state_change`: this is a boolean input indicating if the training is done with state change prediction along with interaction prediction. Default value: True.

### Run the T-Batch code

To create T-Batches of a temporal network, use the following command. This will save a file with T-Batches in the `results/tbatches_<network>.csv` file. Note that the entire input will be converted to T-Batches. To convert only training data, please input a file with only the training interactions. 

```
   $ python tbatch.py --network reddit
```

This code can be given the following command-line arguments:
1. `--network`: this is the name of the file which has the data in the `data/` directory. The file should be named `<network>.csv`, where `<network> = reddit` in the example above. The dataset format is explained below. This is a required argument. 


### Run the evaluation code

To evaluate the trained model's performance in predicting interactions, use the following command. 
```
    $ python evaluate_interaction_prediction.py --network reddit --model jodie --epoch 49
```

To evaluate the trained model's performance in predicting interactions, use the following command. This will add the performance numbers to the `results/interaction_prediction_<network>.txt` file.
```
    $ python evaluate_interaction_prediction.py --network reddit --model jodie --epoch 49
```

To evaluate the trained model's performance in predicting user state change, use the following command. This will add the performance numbers to the `results/state_change_prediction_<network>.txt` file.
```
   $ python evaluate_state_change_prediction.py --network reddit --model jodie --epoch 49
```

### Dataset format

The networks are stored under the `data/` folder, one file per network. The filename should be `<network>.csv`.

The network should be in the following format:
- One line per interaction/edge.
- Each line should be: *user, item, timestamp, state label, comma-separated array of features*.
- First line is the network format. 
- *User* and *item* fields can be alphanumeric.
- *Timestamp* should be in cardinal format (not in datetime).
- *State label* should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
- *Feature list* can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions.

For example, the first few lines of a dataset can be:
```
user,item,timestamp,state_label,comma_separated_list_of_features
0,0,0.0,0,0.1,0.3,10.7
2,1,6.0,0,0.2,0.4,0.6
5,0,41.0,0,0.1,15.0,0.6
3,2,49.0,1,100.7,0.8,0.9
```

### References 
*Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks*. Srijan Kumar, Xikun Zhang, Jure Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 

If you make use of this code, the JODIE algorithm, the T-batch algorithm, or the datasets in your work, please cite the following paper:
```       	    
   @inproceedings{kumar2019predicting,
	 title={Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks},
	 author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},
	 booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},
	 year={2019},
	 organization={ACM}
	}
```