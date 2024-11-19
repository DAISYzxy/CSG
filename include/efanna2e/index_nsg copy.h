#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>
#include <map>

namespace efanna2e {

class IndexNSG : public Index {
 public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexNSG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;

//   virtual void Build(size_t n, const float *data, const Parameters &parameters) override;


  virtual void Build(size_t dimension, size_t n, const float *data, const Parameters &parameters) override;
  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices);
  void OptimizeGraph(float* data);
  void CrossOptimizeGraph(float* data_load1, float* data_load2, const float * query, unsigned num, unsigned num2, unsigned query_num, size_t K, const Parameters &parameters, std::vector<std::vector<std::vector<unsigned>>> &indices, unsigned dim);

  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<std::vector<unsigned>> Clusters;
    typedef std::vector<double> ValueGraph;
    typedef std::vector<float> ValueDiversity;
    typedef std::vector<unsigned> LabelGraph;
    typedef std::map<unsigned, std::map<unsigned, std::vector<unsigned >>> CrossNeighobrs;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;
    typedef std::map<unsigned, std::map<unsigned, std::vector<unsigned >>> SplitTrees;
    typedef std::vector<SplitTrees> SplitForest;

    CompactGraph final_graph_;
    SplitTrees split_trees_;
    ValueGraph pagerank_graph_;
    Clusters cluster_graph_;
    LabelGraph node_cluster_;
    ValueDiversity node_diversity_;
    SplitForest split_forest_;
    CrossNeighobrs cross_links_;


    Index *initializer_ = nullptr;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    // void LoadMap(const char *filename);
    // void SaveMap(const char *filename);
    void Print_nn_graph();
    // void tree_grow(const Parameters &parameter);
    void tree_grow(const Parameters &parameter, unsigned root, CompactGraph &backup_fg);
    // void findroot2(std::map<unsigned, std::vector<unsigned >> &graph, std::vector<unsigned> in_seed, std::map<unsigned, unsigned> &flag, unsigned &root, const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    // void DFS2(std::map<unsigned, std::vector<unsigned>> tmp_graph, std::vector<unsigned> in_seed, std::map<unsigned, unsigned> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);
    void calculatePageRank(double damping_factor, int max_iterations, double tolerance);
    void kMeans(const float* data, unsigned num, unsigned dim, unsigned k);
    void valueAssign(const float* data, unsigned num, unsigned dim);
    float neighbor_union_value(std::vector<unsigned> currentSet);
    void greedyAdd(std::vector<unsigned>& targetSet, size_t candidateSize, size_t budget);
    void inshardBound(std::map<unsigned, std::vector<unsigned >> &graph, std::vector<unsigned> inseed_nodes, unsigned seed, unsigned dim, int B);
    void crossShardBound(const float* data_load1, const float* data_load2, unsigned dim, int Bound);
    // void cut(std::map<unsigned, std::vector<unsigned>> &tmp_graph, std::map<unsigned, std::vector<unsigned>> &new_graph, unsigned root, unsigned target);
    // void mRootedForest(const Parameters &parameter, std::vector<unsigned> seedSet, unsigned root, unsigned dim);
    // void find_parent(std::map<unsigned, std::vector<unsigned>> graph, unsigned root, unsigned target, unsigned &parent);
    void OptimizeGraph2(const float *data, SplitTrees shard, unsigned dimension, unsigned width, size_t total_num);
    void updateRetset(std::vector<Neighbor>& tmp, unsigned int K, std::vector<Neighbor>& retset);
    std::vector<Neighbor> SearchSubGraphOpt(std::map<unsigned, std::vector<unsigned >> tmp_graph, unsigned ep, const float *data, const float *query, efanna2e::Distance* dist, char* opt_graph, unsigned ns, unsigned dl, size_t K,
                                  const Parameters &parameters, unsigned dim);
    
  private:
    unsigned width;
    unsigned ep_;
    unsigned Bound;
    std::vector<std::mutex> locks;
    char* opt_graph_ = nullptr;
    char* opt_graph2_ = nullptr;
    size_t node_size = 0;
    size_t data_len = 0;
    size_t neighbor_len = 0;
    size_t node_size2 = 0;
    size_t data_len2 = 0;
    size_t neighbor_len2 = 0;
    KNNGraph nnd_graph;
};
}

#endif //EFANNA2E_INDEX_NSG_H
