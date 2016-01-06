This example expects that the data preparation portions of the cifar10 example has already been done.

From the caffe root directory, run this example across three nodes using your systems mpirun/aprun command.
mpirun -np 3 caffe train --solver=examples/cifar10-mpi/cifar10_mpi_solver.prototxt 
