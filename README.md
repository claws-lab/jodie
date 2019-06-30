# JODIE
Code for ACM SIGKDD 2019 paper "Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks"

Link to paper: https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf

Project website with links to datasets: http://snap.stanford.edu/jodie/


# Introduction
JODIE is a representation learning framework for temporal interaction networks. Given a sequence of entity-entity interactions, JODIE learns a dynamic embedding trajectory for every entity, which can then be used for various downstream machine learning tasks. JODIE is fast and makes accurate predictions on temporal interaction network.

# Motivation 
Temporal interaction networks provide an expressive language to represent time-evolving and dynamic interactions between entities. Representation learning provides a powerful tool to model and reason on networks. However, as networks evolve over time, a single (static) embedding becomes insufficient to represent the changing behavior of the entities and the dynamics of the network.

JODIE is a representation learning framework that embeds every entity in a Euclidean space and their evolution is modeled by an embedding trajectory in this space. JODIE learns to project/forecast the embedding trajectories into the future to make predictions about the entities and their interactions. These trajectories can be trained for downstream tasks, such as recommendations and predictions. JODIE is scalable to large networks by employing a novel t-Batch algorithm that creates batches of independent edges that can be processed simulaneously.

# References 
Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 

You may use the following BibTeX entry:

@inproceedings{kumar2019predicting,

title={Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks},

author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},

booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},

year={2019},

organization={ACM}

}
