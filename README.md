[This code is no longer supported and way behind the times at this point.]

mpi-caffe combines the modularity of the popular [Caffe](http://caffe.berkeleyvision.org) deep learning framework with the powerful Message Passing Interface (MPI) standard -- enabling simple, modular design of deep networks that span multiple machines! 

Communication logic in mpi-caffe is abstracted to model layers which are included as part of the network architecture, allowing researchers to quickly experiment with model-distributed networks in environments ranging from a multiple GPU workstation to a large-scale cluster. With mpi-caffe it is easy to experiment with extremely large networks which don't fit into a single node's memory, link multiple full networks under a unified loss, or allow a set of networks to share some common computation. 



