# CSG: Multimodal Approximate Nearest Neighbor Search with The Cross-Shards Graph

<img src="case_study.png" width="1000px">

We propose a novel Cross-Shards Graph (CSG) to address issues in current Multimodal Approximate Neareast Neighbor Search (ANNS). We define the search entry selection problem as a reduction of the maximum independent set problem and introduce a greedy algorithm with theoretical guarantees as the solution. We also establish cross-shards connectivity through a two-phase procedure, ensuring that the search complexity remains approximately $\mathcal{O}(\log N)$ and providing a theoretical proof of the reduction in the expected search path length. Our extensive experiments demonstrate that our algorithm surpasses current state-of-the-art ANNS algorithms in terms of search accuracy and efficiency on the real-world multimodal datasets.


Table of Contents
=================
<!--ts-->
* [Introduction](#introduction)
* [Building Instruction](#building-instruction)
     * [Prerequisites](#prerequisites)
     * [Compile On Ubuntu/Debian](#compile-on-ubuntudebian)
* [Usage](#usage)
     * [Building CSG Index](#building-csg-index)
     * [Searching via CSG Index](#searching-via-csg-index)
     * [Datasets](#datasets)

## Building Instruction

### Prerequisites

+ GCC 4.9+ with OpenMP
+ CMake 2.8+
+ Boost 1.55+
+ [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html)


### Compile On Ubuntu/Debian

1. Install Dependencies:

```shell
$ sudo apt-get install g++ cmake libboost-dev libgoogle-perftools-dev
```

2. Compile CSG:

```shell
$ git clone https://github.com/DAISYzxy/CSG.git
$ cd CSG/
$ mkdir build/ && cd build/
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j
```


## Usage

The main interfaces and classes have its respective test codes under directory `tests/`

### Building CSG Index


#### Step 1. Build a KNN graph

Plese refer to [efanna\_graph](https://github.com/ZJULearning/efanna\_graph) to build the kNN graph. Otherwise you can also use any alternatives, such as KGraph or faiss.

#### Step 2. Construct a CSG index with Seed Nodes

Secondly, we will construct a CSG index with seed nodes as follows:

```shell
$ cd build/tests/
$ ./test_seed_points DATA_PATH KNNG_PATH L R C SEEDS_PATH CSG_PATH
```

+ `DATA_PATH` is the path of the base data in `fvecs` format.
+ `KNNG_PATH` is the path of the pre-built kNN graph in *Step 1.*.
+ `L, R, C` follows the same meaning in [NSG](https://github.com/ZJULearning/nsg) which CSG refines on.
+ `SEEDS_PATH` is the path of the generated seeds data.
+ `CSG_PATH` is the path of the generated CSG index.

### Searching via CSG Index

Here are the instructions of how to use CSG index for multimodal retrieval.

```shell
$ cd build/tests/
$ ./test_searching MODAL1_PATH MODAL2_PATH QUERY_PATH MODAL1_CSG_PATH MODAL2_CSG_PATH MODAL1_SEEDS_PATH MODAL2_SEEDS_PATH QUERY_MODAL1_GT QUERY_MODAL2_GT SEARCH_L SEARCH_K
```

+ `MODALx_PATH` is the path of the x-th modal base data in `fvecs` format.
+ `QUERY_PATH` is the path of the query data in `fvecs` format.
+ `MODALx_CSG_PATH` is the path of the pre-built CSG index for x-th modal base data in previous section.
+ `MODALx_SEEDS_PATH` is the path of the pre-selected seeds for x-th modal base data in previous section.
+ `MODALx_SEEDS_PATH` is the path of ground truth for querying in x-th modal base data in `ivecs` format.
+ `SEARCH_L, SEARCH_K` follows the same meanings in [NSG](https://github.com/ZJULearning/nsg).

### Datasets
To test the performance of CSG, please download the base data, ground truth, pre-built CSG index and seeds in the [Drive](https://drive.google.com/file/d/1A16uRDzC9PIF8vs9pdwNdGs8MQPEQj5b/view?usp=sharing).

***NOTE:** Only data-type int32 and float32 are supported for now.*
