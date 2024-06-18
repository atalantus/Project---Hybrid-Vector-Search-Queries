# Project: Hybrid Vector Search Queries
This project is about optimizing the baseline implementation (`baseline.cpp`) of a vector search engine. Therefore, you should use the techniques and methods we've shown in the lecture (i.e., vectorization, I/O optimization, cache awareness, etc.).

## Description
 Given a set of vectors with additional attributes, the task is to answer hybrid vector search queries over the data accurately as fast as possible. A hybrid vector query is to find the approximate **k nearest neighbors** of a given query vector under one given similarity measure, such as *Euclidean distance*, with some constraints on non-vector attributes. For each query, your output should be the ids of the **k** nearest neighbors determined by your algorithm. 
 
 - **k** is set to 100
 - vectors have a dimension of 100
 
The sample dataset (referred to as **D**) contains millions of high-dimensional vectors, each also having a discretized categorical attribute (denoted as **C**) and a normalized timestamp attribute (denoted as **T**). The query set (referred to as **Q**) contains millions of hybrid vector queries.

### Data Set
Dataset D is in a binary format, beginning with a 4 byte integer num_vectors (`uint32_t`) indicating the number of vectors. This is followed by data for each vector, stored consecutively, with each vector occupying *102* `(2 + vector_num_dimension) x sizeof(float32)` bytes, summing up to `num_vectors x 102 (2 + vector_num_dimension) x sizeof(float32)` bytes in total.
Specifically, for the 102 dimensions of each vector: the first dimension denotes the discretized categorical attribute C and the second dimension denotes the normalized timestamp attribute T. The rest 100 dimensions are the vector.

### Query Set
 Query set Q is in a binary format, beginning with a 4-byte integer num_queries (`uint32_t`) indicating the number of queries. This is followed by data for each query, stored consecutively, with each query occupying 104 `(4 + vector_num_dimension) x sizeof(float32)` bytes, summing up to `num_queries x 104 (4 + vector_num_dimension) x sizeof(float32)` bytes in total.

The 104-dimensional representation for a query is organized as follows:

* The first dimension denotes query_type (takes values from 0, 1, 2, 3).
* The second dimension denotes the specific query value v for the categorical attribute (if not queried, takes -1).
* The third dimension denotes the specific query value l for the timestamp attribute (if not queried, takes -1).
* The fourth dimension denotes the specific query value r for the timestamp attribute (if not queried, takes -1).
* The rest 100 dimensions are the query vector.

There are four types of queries, i.e., the query_type takes values from 0, 1, 2 and 3. The 4 types of queries correspond to:

* If query_type=0: Vector-only query, i.e., the conventional approximate nearest neighbor (ANN) search query.
*If query_type=1: Vector query with categorical attribute constraint, i.e., ANN search for data points satisfying `C=v`.
* If query_type=2: Vector query with timestamp attribute constraint, i.e., ANN search for data points satisfying `l≤T≤r`.
* If query_type=3: Vector query with both categorical and timestamp attribute constraints, i.e. ANN (Approximate Nearest Neighbour) search for data points satisfying `C=v` and `l≤T≤r`.

The predicate for the categorical attribute is an equality predicate ( i.e., `C=v`). And the predicate for the timestamp attribute is a range predicate (i.e., `l≤T≤r`).

### Input
Functions for reading and writing Q and D are provided in the file `io.h`. There are multiple data and query sets with different sizes.

* Default/provided: **D** (10^4) **Q** (10^2)
* Medium size: [**D**](https://contestdata.blob.core.windows.net/sigmoddata/contest-data-release-1m.bin) (10^6) [**Q**](https://contestdata.blob.core.windows.net/sigmoddata/contest-queries-release-1m.bin) (10^4)
* Large size: [**D**](https://contestdata.blob.core.windows.net/sigmoddata/contest-data-release-10m.bin) (10^7) [**Q**](https://contestdata.blob.core.windows.net/sigmoddata/Public-4M-queries.bin) (4 x 10^4)

### Output
Your output should be the ids of the k nearest neighbors determined by your algorithm. These neighbor lists are stored one by one and stored in a **binary file**: `output.bin`

`output.bin`: The file should contain `|Q| x 100 x id` (uint32_t). `|Q|` is the number of queries in query set Q, 100 is the number of nearest neighbors and id is the index of 100 nearest neighbors in the given dataset D. 

### Note
It is **prohibitive** to use query vectors during the indexing phase.

## Compilation
Build the project
```
make
```
Build and run the project
```
chmod +x run.sh
./run.sh
```
Run the project
```
./test
```

## Organization
- This is not a group project: you're working on your own
- You have 4 weeks for the project
- You have to present and **explain** your results and doing in a 5-minute presentation
- You can achieve **5 points** bonus 

### Copyright
This project was part of the *ACM SIGMOD Programming Contest*, created by the DB Group at *Tsinghua University* and the DB Group at *Rutgers University*.