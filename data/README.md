# Data description

For all problem classes, data is pickled. 

## TSP

Pickled data is a **list** of instances. Each instance is a **dictionary** with the following keys:

- `inst`: Numpy array of shape `(<num nodes>, 2)`, containing the two-dimensional coordinates for each node.
- `tour`: The solution given as a list of integers (length `<num nodes>`), which is a permutation of node indices (from 1 to `<num_nodes>`).
- `sol`: Tour length of the given solution.

## CVRP

Pickled data is a **list** of instances. Each instance is a **dictionary** with the following keys:

- `capacity`: (float) Capacity of the vehicle.
- `nodes`: Numpy array of shape `(1 + <num customers>, 2)` containing the coordinates of depot (first entry) and customers.
- `demands`: Numpy array of shape `(1 + <num customers>)` containing the demand of each customer, where first entry is 0 ('demand of depot').
- `tour_length`: Total tour length of the given solution.
- `tour_node_idcs`: Part of the solution. See Appendix on CVRP of the paper for how solutions are structured. List of length `<num customers>` which is a permutation of customer indices. Indexing starts from 1.
- `tour_node_flags`: List of length `<num customers>` where each entry is a 1 if the customer is reached via the depot in the solution, or a 0 if it is reached directly from previous customer. Note that the first entry in the list is always 1, as the first customer is always reached via the depot.

## JSSP

Pickled data is a **list** of instances. Each instance is a **dictionary** with the following keys. Denote by `J` the number of jobs and by `M` the number of machines and by `O=M` the number of operations.

- `proc_times`: Numpy array of shape `(J, O)` with the processing time for each operation. Processing time is an integer between 1 and 99.
- `ops_machines`: Numpy array of shape `(J, O)` with the index of the machine on which an operation must run.
- `title`: Optional title of the instance to identify it (used for instances from the literature).
- `obj`: Makespan of the solution as integer.
- `job_seq`: Solution of the given instance, represented by the
ordered sequence of jobs whose next operation should be scheduled. See Appendix on JSSP in the paper for a description on the structure of solutions.
- `op_mask_by_machines`: Pre-generated boolean numpy array of shape (`J*O`, `J*O`) which can serve as an attention mask where an operation can only attend to another operation if it runs on the same machine. In the array, the `i, j`-th value is `False` if and only if operation `i` and `j` run on the same machine. Here, indexing is done as follows: The first `O` operations belong to the first job, operations `O+1,..., 2*O` belong to the second job and so on.