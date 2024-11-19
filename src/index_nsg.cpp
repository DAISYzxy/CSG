#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <vector>
#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include <map>
#include <float.h>
#include <queue>
#include <functional>
#include <unordered_set>
#include <future>
#include <mutex>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"


float euclideanDistance(const float* a, const float* b, unsigned dim) {
    float distance = 0.0;
    for (unsigned i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}


float cosineSimilarity(const float* a, const float* b, unsigned dim) {
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    for (unsigned i = 0; i < dim; ++i) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dotProduct / (std::sqrt(normA) * std::sqrt(normB));
}


float arccosDistance(const float* a, const float* b, unsigned dim) {
    float similarity = cosineSimilarity(a, b, dim);
    similarity = std::max(-1.0f, std::min(1.0f, similarity));
    return std::acos(similarity);
}


std::set<unsigned> getIntersectionForNode(
    std::vector<unsigned> clusterNodes,
    std::vector<unsigned> neighbors) {
    std::set<unsigned> clusterNodeSet(clusterNodes.begin(), clusterNodes.end());
    std::set<unsigned> neighborSet(neighbors.begin(), neighbors.end());
    std::set<unsigned> intersection;
    std::set_intersection(
        clusterNodeSet.begin(), clusterNodeSet.end(),
        neighborSet.begin(), neighborSet.end(),
        std::inserter(intersection, intersection.begin())
    );
    return intersection;
}

void saveToIvecs(const char *filename, const std::vector<unsigned>& seed_points) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Could not open file for writing.\n";
        return;
    }

    // Write the size of the vector
    unsigned size = seed_points.size();
    outfile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Write the vector elements
    for (unsigned value : seed_points) {
        outfile.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    outfile.close();
}


float boundDiversity(
    unsigned nodeIndex,
    const std::set<unsigned>& nodeSet,
    const std::set<unsigned>& nodeSet2, // difference set
    const float* data,
    unsigned num,
    unsigned dim) {
    
    float max_v = 0.0f;
    size_t set_len = nodeSet2.size();
    for (unsigned node1 : nodeSet) {
        if (node1 >= num) {
            throw std::out_of_range("Node index is out of range.");
        }
        const float* nodeVector1 = data + node1 * dim;
        for (unsigned node2 : nodeSet2){
          if (node2 >= num) {
              throw std::out_of_range("Node index is out of range.");
          }
          const float* nodeVector2 = data + node2 * dim;
          float dis = arccosDistance(nodeVector1, nodeVector2, dim);
          if (dis > max_v) max_v = dis;
        }
    }
    return max_v * set_len;
}


// std::vector<unsigned> nearestOrBounded(const float *data, const float * query, std::vector<unsigned> seeds, unsigned dimension, unsigned bound){
//   unsigned id = 0;
//   std::vector<unsigned> result;
//   float min_dis = FLT_MAX;
//   unsigned min_seed = 0;
//   for(int i = 0; i < dimension; i++) std::cout << query[i] << ", ";
//   std::cout << std::endl;
//   for (int i = 0; i < seeds.size(); i++){
//     id = seeds[i];
//     std::cout << data << std::endl;
//     // for (unsigned i = 0; i < dimension; i++) std::cout << data[id * dimension + i] << ", ";
//     // std::cout << std::endl;
//     float dis = euclideanDistance(query, data + id * dimension, dimension);
//     std::cout << dis << std::endl;
//     if (dis < bound) result.push_back(id);
//     if (dis < min_dis){
//       min_dis = dis;
//       min_seed = id;
//     }
//   }
//   if (std::find(result.begin(), result.end(), min_seed) == result.end()){
//     result.push_back(min_seed);
//   }
//   // result.push_back(min_seed);
//   return result;
// }


std::map<unsigned, std::vector<unsigned >> convertGraph(std::vector<std::vector<unsigned >> graph) {
    std::map<unsigned, std::vector<unsigned >> graph_map;
    for (unsigned i = 0; i < graph.size(); ++i) {
        graph_map[i] = graph[i];
    }
    return graph_map;
}

namespace efanna2e {
#define _CONTROL_NUM 100
IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                   Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}

IndexNSG::~IndexNSG() {
    if (distance_ != nullptr) {
        delete distance_;
        distance_ = nullptr;
    }
    if (initializer_ != nullptr) {
        delete initializer_;
        initializer_ = nullptr;
    }
    if (opt_graph_ != nullptr) {
        delete opt_graph_;
        opt_graph_ = nullptr;
    }
    if (distance2_ != nullptr) {
        delete distance2_;
        distance2_ = nullptr;
    }
    if (opt_graph2_ != nullptr) {
        delete opt_graph2_;
        opt_graph2_ = nullptr;
    }
}

// std::vector<std::vector<Neighbor>> IndexNSG::SearchParallel(
//   const std::vector<unsigned>& seed_entries,
//   std::map<unsigned, std::map<unsigned, std::vector<unsigned>>> &tmp_graph,
//   const float *data,
//   const float *query,
//   efanna2e::Distance *dist,
//   char* opt_graph, unsigned ns, unsigned dl, size_t K,
//   const Parameters &parameters,
//   unsigned dim) {

//   std::vector<std::future<std::vector<Neighbor>>> futures;
//   for (unsigned ep : seed_entries) {
//       // Launch a task for each entry point
//       futures.push_back(std::async(std::launch::async,
//                                     &IndexNSG::SearchSubGraphOpt,
//                                     this,  // Assuming this is a member function of IndexNSG. Adjust `this` accordingly if it's not.
//                                     tmp_graph[ep],
//                                     ep,
//                                     data,
//                                     query,
//                                     dist,
//                                     opt_graph,
//                                     ns,
//                                     dl,
//                                     K,
//                                     parameters,
//                                     dim));
//   }

//   std::vector<std::vector<Neighbor>> results;
//   for (auto& future : futures) {
//       results.push_back(future.get());
//   }

//   return results;
// }

// void IndexNSG::Save(const char *filename) {
//   std::ofstream outFile(filename);
//   if (outFile.is_open()) {
//       for (const auto& pair : split_trees_) {
//           unsigned key = pair.first;
//           const auto& submap = pair.second;
//           // Save the key for the main map
//           outFile << key << " " << submap.size() << "\n";
//           for (const auto& subpair : submap) {
//               unsigned subkey = subpair.first;
//               const auto& vec = subpair.second;
//               // Save the key for the sub-map and the size of the vector
//               outFile << subkey << " " << vec.size() << " ";
//               // Save each element of the vector
//               for (unsigned int val : vec) {
//                   outFile << val << " ";
//               }
//               outFile << "\n";
//           }
//       }
//       outFile.close();
//   } else {
//       std::cerr << "Unable to open file for saving.\n";
//   }
// }




// void IndexNSG::Load(const char *filename) {
//   split_trees_.clear(); // Clear the map before loading new contents
//   std::ifstream inFile(filename);
//   if (inFile.is_open()) {
//       unsigned key, subkey, vecSize, submapSize;
//       while(inFile >> key >> submapSize) {
//           std::map<unsigned, std::vector<unsigned>> submap;
//           for (unsigned i = 0; i < submapSize; ++i) {
//               inFile >> subkey >> vecSize;
//               std::vector<unsigned> vec(vecSize);
//               for (unsigned j = 0; j < vecSize; ++j) {
//                   inFile >> vec[j];
//               }
//               submap.insert({subkey, vec});
//           }
//           split_trees_.insert({key, submap});
//       }
//       inFile.close();
//   } else {
//       std::cerr << "Unable to open file for loading.\n";
//   }
//   split_forest_.push_back(split_trees_);
//   std::cout << "# Seeds: " << split_trees_.size() << std::endl;
// }

void IndexNSG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void IndexNSG::Load(const char *filename) {
  final_graph_.clear();
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));
  // width=100;
  unsigned cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
  split_forest_.push_back(final_graph_);
  width_vec_.push_back(width);
  // std::cout<<cc<<std::endl;
}


void IndexNSG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();
}


void IndexNSG::Print_nn_graph() {
  for (size_t i = 0; i < final_graph_.size(); ++i) {
    std::cout << "Node " << i << ": ";
    for (size_t j = 0; j < final_graph_[i].size(); ++j) {
      std::cout << final_graph_[i][j];
      if (j < final_graph_[i].size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
}

void IndexNSG::calculatePageRank(double damping_factor = 0.85, int max_iterations = 100, double tolerance = 1e-6) {
    size_t num_nodes = final_graph_.size();
    std::vector<double> page_rank(num_nodes, 1.0 / num_nodes);
    std::vector<double> temp_page_rank(num_nodes, 0.0);

    for (int iter = 0; iter < max_iterations; ++iter) {
        double dangling_sum = 0.0; // Sum of PageRank of dangling nodes

        // Compute the contribution from each node
        for (size_t i = 0; i < num_nodes; ++i) {
            temp_page_rank[i] = 0.0;
            for (unsigned neighbor : final_graph_[i]) {
                temp_page_rank[neighbor] += page_rank[i] / final_graph_[i].size();
            }
            if (final_graph_[i].empty()) {
                dangling_sum += page_rank[i];
            }
        }

        // Apply the damping factor and distribute the dangling sum
        double one_minus_damping = (1.0 - damping_factor) / num_nodes;
        dangling_sum *= damping_factor / num_nodes;

        // Update PageRank with damping factor, dangling nodes and normalize
        double diff = 0.0;
        for (size_t i = 0; i < num_nodes; ++i) {
            temp_page_rank[i] = damping_factor * temp_page_rank[i] + one_minus_damping + dangling_sum;
            diff += std::abs(temp_page_rank[i] - page_rank[i]);
        }

        // Check for convergence
        if (diff < tolerance) {
            break;
        }

        std::swap(page_rank, temp_page_rank);
    }
    // normalize the pagerank value to 0 - 180
    double max = *std::max_element(page_rank.begin(), page_rank.end());
    double min = *std::min_element(page_rank.begin(), page_rank.end());
    for (size_t i = 0; i < num_nodes; ++i) {
        page_rank[i] = (page_rank[i] - min) / (max - min) * 180;
    }
    pagerank_graph_ = page_rank;
}

void IndexNSG::kMeans(const float* data, unsigned num, unsigned dim, unsigned k) {
    std::vector<unsigned> labels(num, 0);
    std::vector<float> centroids(k * dim, 0.0f);
    std::vector<unsigned> counts(k, 0);
    std::vector<std::vector<unsigned>> clusters(k);
    bool changed = true;

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, num - 1);
    for (unsigned i = 0; i < k; ++i) {
        unsigned dataIndex = distribution(generator);
        std::copy(data + dataIndex * dim, data + (dataIndex + 1) * dim, centroids.begin() + i * dim);
    }
    while (changed) {
        changed = false;
        // Clear previous clustering information
        for (auto& cluster : clusters) {
            cluster.clear();
        }
        for (unsigned i = 0; i < num; ++i) {
            float minDistance = std::numeric_limits<float>::max();
            unsigned bestCluster = 0;
            for (unsigned j = 0; j < k; ++j) {
                float distance = euclideanDistance(data + i * dim, &centroids[j * dim], dim);
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = j;
                }
            }
            if (labels[i] != bestCluster) {
                labels[i] = bestCluster;
                changed = true;
            }
            clusters[bestCluster].push_back(i);
        }

        std::fill(centroids.begin(), centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (unsigned i = 0; i < num; ++i) {
            unsigned cluster = labels[i];
            for (unsigned j = 0; j < dim; ++j) {
                centroids[cluster * dim + j] += data[i * dim + j];
            }
            counts[cluster]++;
        }
        for (unsigned i = 0; i < k; ++i) {
            if (counts[i] == 0) continue; // Avoid division by zero
            for (unsigned j = 0; j < dim; ++j) {
                centroids[i * dim + j] /= counts[i];
            }
        }
    }
    // for (unsigned i = 0; i < clusters.size(); ++i) {
    //   std::cout << "Cluster " << i << ": ";
    //   for (unsigned nodeIndex : clusters[i]) {
    //       std::cout << nodeIndex << " ";
    //   }
    //   std::cout << std::endl;
    // }
    node_cluster_ = labels;  // labels[i] is the cluster label for node i
    cluster_graph_ = clusters;  // clusters[i] contains all nodes in cluster i
}


void IndexNSG::valueAssign(const float* data, unsigned num, unsigned dim) {
    std::vector<float> values(num, 0.0f);
    for (unsigned i = 0; i < num; ++i) {
        unsigned clusterIndex = node_cluster_[i];
        std::vector<unsigned> clusterNodes = cluster_graph_[clusterIndex];
        std::vector<unsigned> neighbors = final_graph_[i];
        std::set<unsigned> intersection = getIntersectionForNode(clusterNodes, neighbors);
        std::set<unsigned> neighborSet(neighbors.begin(), neighbors.end());
        std::set<unsigned> difference;
        std::set_difference(
            neighborSet.begin(), neighborSet.end(),
            intersection.begin(), intersection.end(),
            std::inserter(difference, difference.begin())
        );
        values[i] = boundDiversity(i, intersection, difference, data, num, dim);
        // std::cout << "Node " << i << ": " << values[i] << std::endl;
    }
    node_diversity_ = values;

}


void IndexNSG::greedyAdd(std::vector<unsigned>& targetSet, size_t candidateSize, size_t budget) {
  for (unsigned v = 0; v < budget; ++v){
    float currentGain = neighbor_union_value(targetSet);
    for (unsigned j = 0; j < targetSet.size(); ++j){
      currentGain += node_diversity_[j];
    }
    // std::cout << "Current Target Set Value: " << currentGain << std::endl;
    std::vector<float> gains(candidateSize, 0.0f);
    for (unsigned i = 0; i < candidateSize; ++i) {
      std::vector<unsigned> clusterNodes = targetSet;
      clusterNodes.push_back(i);
      float pagerank_value = neighbor_union_value(clusterNodes);
      gains[i] = pagerank_value;
      for (unsigned j = 0; j < clusterNodes.size(); ++j){
        gains[i] += node_diversity_[j];
      }
    }
    unsigned maxIndex = 0;
    float maxDelta = 0.0f;
    for (unsigned i = 0; i < gains.size(); ++i) {
      float delta = gains[i] - currentGain;
      if (delta > maxDelta) {
          maxDelta = delta;
          maxIndex = i;
      }
    }
    targetSet.push_back(maxIndex);
  }
}



float IndexNSG::neighbor_union_value(std::vector<unsigned> currentSet) {
    float total = 0.0f;
    std::set<unsigned> unionSet;
    for (unsigned i = 0; i < currentSet.size(); ++i) {
        unsigned nodeIndex = currentSet[i];
        std::vector<unsigned> neighbors = final_graph_[nodeIndex];
        for (unsigned neighbor : neighbors) {
          unionSet.insert(neighbor);
        }
    }
    for (unsigned neighbor : unionSet) {
      total += pagerank_graph_[neighbor];
    }
    return total;
}



void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

// 函数结束时，retset包含了距离查询点最近的L个邻居，fullset则包含了所有考虑过的邻居，包括被排序和剪枝过程中排除的邻居。
// flags是一个动态位集，用来记录已经被访问的点以避免重复处理
// distance_是一个距离计算对象，data_是数据点的数组，dimension_是数据点的维度。
void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");
  // 调整retset的大小为L + 1，并创建一个初始ID列表init_ids，大小也为L
  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  // 将L重置为0。遍历init_ids和final_graph_[ep_]（ep_是起始点的索引）：
    // 将final_graph_[ep_]中的ID复制到init_ids中。
    // 在flags中将对应的标志位置为true，表示已被访问。
    // 增加L的值。
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  // 如果L小于init_ids.size()，则在数据集中随机选择一个未被访问的点，加入到init_ids中，并在flags中标记为已访问。
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  // L再次重置为0。遍历init_ids中的每个ID：
    // 如果ID大于等于数据集大小nd_，则跳过。
    // 计算查询点query与数据点的距离。
    // 将该数据点作为Neighbor对象存入retset和fullset。
    // 增加L的值。
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    fullset.push_back(retset[i]);
    // flags[id] = 1;
    L++;
  }
  // 对retset前L个元素排序
  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  ep_ = rand() % nd_;  // random initialize navigating point
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id;
  delete center;
}



// 这个函数的主要目标是为每个查询点q找到一组近邻，并且确保MRNG的选边策略。
// 在这个过程中，使用了动态位集flags来标记已访问过的点，以避免重复计算。
// 此外，distance_是一个用于计算两点间距离的对象，data_是存储所有数据点的数组，dimension_是数据的维度。最终，cut_graph_会被更新为包含修剪后的近邻列表。
void IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                          const Parameters &parameter,
                          boost::dynamic_bitset<> &flags,
                          SimpleNeighbor *cut_graph_) {

  // 从参数parameter中获取范围range和最大候选数量maxc。
  unsigned range = parameter.Get<unsigned>("R");
  unsigned maxc = parameter.Get<unsigned>("C");

  // 设置width为range，start为0，用来在后面的步骤中追踪当前的起始位置。
  width = range;
  unsigned start = 0;

  // 遍历查询点q在final_graph_中的所有邻居：
    // 如果当前邻居的标志位在flags中已被设置（即已访问），则跳过。
    // 否则，计算查询点q与当前邻居的距离，并将邻居加入到pool中。
  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id]) continue;
    float dist =
        distance_->compare(data_ + dimension_ * (size_t)q,
                           data_ + dimension_ * (size_t)id, (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }
  // 对pool中的所有邻居按照距离进行排序
  std::sort(pool.begin(), pool.end());

  // 创建一个新的向量result，用来存储最终的结果
  std::vector<Neighbor> result;

  // 如果pool中的第一个元素（距离最小的邻居）是查询点本身，则跳过这个点。
  if (pool[start].id == q) start++;

  // 将pool中的第一个邻居（或第二个如果第一个是查询点本身）加入到result中
  result.push_back(pool[start]);

  // 在pool中继续向后查找，直到结果集合result的大小达到range或者遍历完maxc个候选邻居：
  while (result.size() < range && (++start) < pool.size() && start < maxc) {
    auto &p = pool[start];
    bool occlude = false;
    // 对于每个候选邻居，检查是否被结果集合中已有的点遮挡。
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);
      if (djk < p.distance /* dik */) {
        occlude = true;
        break;
      }
    }
    // 如果没有被遮挡，则将其加入到result中。
    if (!occlude) result.push_back(p);
  }

  // 将result中的邻居信息复制到cut_graph_中对应查询点q的位置。
  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); t++) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  // 如果result的大小小于range，则在cut_graph_中设置一个标记值-1来表示结束。
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
}

void IndexNSG::InterInsert(unsigned n, unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1) break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup) continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                         data_ + dimension_ * (size_t)p.id,
                                         (unsigned)dimension_);
          if (djk < p.distance /* dik */) {
            occlude = true;
            break;
          }
        }
        if (!occlude) result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range) des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}
// 这个函数使用了OpenMP来并行化处理
// cut_graph_：一个指向SimpleNeighbor类型数组的指针，用于存储构建的近邻图
void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
  /*
  std::cout << " graph link" << std::endl;
  unsigned progress=0;
  unsigned percent = 100;
  unsigned step_size = nd_/percent;
  std::mutex progress_lock;
  */
  unsigned range = parameters.Get<unsigned>("R");
  // locks是一个包含nd_个互斥锁的向量，用于在并行插入操作中防止数据竞争
  std::vector<std::mutex> locks(nd_);

// #pragma omp parallel是一个OpenMP指令，它告诉编译器下面的代码块需要并行执行。
#pragma omp parallel
  {
    // unsigned cnt = 0;
    // 在每个并行线程中，定义了两个Neighbor类型的向量pool和tmp，以及一个动态位集flags
    // pool和tmp用于存储邻居节点，而flags用于标记已经处理过的节点。
    std::vector<Neighbor> pool, tmp;
    boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; ++n) {
      pool.clear();
      tmp.clear();
      flags.reset();
      get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
      sync_prune(n, pool, parameters, flags, cut_graph_);
      /*
    cnt++;
    if(cnt % step_size == 0){
      LockGuard g(progress_lock);
      std::cout<<progress++ <<"/"<< percent << " completed" << std::endl;
      }
      */
    }
  }
// 这个循环是第二个并行化的部分，使用InterInsert函数在已有的近邻图中进一步插入节点。
// 这同样采用了动态调度，以提高并行效率。
#pragma omp for schedule(dynamic, 100)
  for (unsigned n = 0; n < nd_; ++n) {
    InterInsert(n, range, locks, cut_graph_);
  }
}

void IndexNSG::Build(size_t dim, size_t n, const float *data, const Parameters &parameters, const char *filename) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R");
  Load_nn_graph(nn_graph_path.c_str());
  calculatePageRank();
  data_ = data;
  kMeans(data_, n, dim, 50);
  valueAssign(data_, n, dim);
  std::vector<unsigned> seedSet;
  greedyAdd(seedSet, n, 40);
  std::sort(seedSet.begin(), seedSet.end());
  init_graph(parameters);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);

  for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) break;
      pool_size = j;
    }
    pool_size++;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }


  // for (size_t i = 0; i < seedSet.size(); i++) {
  //   CompactGraph backup_graph(final_graph_);
  //   tree_grow(parameters, seedSet[i], backup_graph);
  //   std::map<unsigned, std::vector<unsigned>> convert_final_graph;
  //   convert_final_graph = convertGraph(backup_graph);
  //   split_trees_.insert(std::pair<unsigned, std::map<unsigned, std::vector<unsigned>>>(seedSet[i], convert_final_graph));
  // }

  saveToIvecs(filename, seedSet);
  tree_grow(parameters, seedSet[0]);

  has_built = true;
  delete cut_graph_;
}


void IndexNSG::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  unsigned tmp_l = 0;
  for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
    init_ids[tmp_l] = final_graph_[ep_][tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    float dist =
        distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;
        float dist =
            distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                  const Parameters &parameters, unsigned *indices) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned tmp_l = 0;
  // search entry is ep_
  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
  unsigned MaxM_ep = *neighbors;
  neighbors++;

  for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
    init_ids[tmp_l] = neighbors[tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  }
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float *x = (float *)(opt_graph_ + node_size * id);
    float norm_x = *x;
    x++;
    float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
    L++;
  }
  // std::cout<<L<<std::endl;

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
      for (unsigned m = 0; m < MaxM; ++m)
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (flags[id]) continue;
        flags[id] = 1;
        float *data = (float *)(opt_graph_ + node_size * id);
        float norm = *data;
        data++;
        float dist = dist_fast->compare(query, data, norm, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        // if(L+1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

// void IndexNSG::OptimizeGraph(const float *data) {  // use after build or load

//   // data_ = data;
//   data_len = (dimension_ + 1) * sizeof(float);
//   neighbor_len = (width + 1) * sizeof(unsigned);
//   node_size = data_len + neighbor_len;
//   opt_graph_ = (char *)malloc(node_size * nd_);
//   DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
//   for (unsigned i = 0; i < nd_; i++) {
//     char *cur_node_offset = opt_graph_ + i * node_size;
//     float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
//     std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
//     std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
//                 data_len - sizeof(float));

//     cur_node_offset += data_len;
//     unsigned k = final_graph_[i].size();
//     std::memcpy(cur_node_offset, &k, sizeof(unsigned));
//     std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
//                 k * sizeof(unsigned));
//     std::vector<unsigned>().swap(final_graph_[i]);
//   }
//   CompactGraph().swap(final_graph_);
// }


void IndexNSG::updateRetset(std::vector<Neighbor>& tmp, unsigned int K, std::vector<Neighbor>& retset) {
    // Merge tmp and retset
    std::vector<Neighbor> mergedSet;
    mergedSet.insert(mergedSet.end(), retset.begin(), retset.end());
    mergedSet.insert(mergedSet.end(), tmp.begin(), tmp.end());
    if (mergedSet[0].id == 0){mergedSet.erase(mergedSet.begin());}

    // Sort the merged set based on distance
    std::sort(mergedSet.begin(), mergedSet.end());

    // Remove duplicates based on id
    auto last = std::unique(mergedSet.begin(), mergedSet.end(), [](const Neighbor& a, const Neighbor& b) {
        return a.id == b.id;
    });
    // auto zero = std::remove_if(mergedSet.begin(), mergedSet.end(), [](const Neighbor& a) {
    //     return a.id == 0;
    // });
    mergedSet.erase(last, mergedSet.end());
    // mergedSet.erase(zero, mergedSet.end());

    // Update retset with the top K elements from the merged set
    std::partial_sort(mergedSet.begin(), mergedSet.begin() + K, mergedSet.end());
    retset.assign(mergedSet.begin(), mergedSet.begin() + K);
}



void IndexNSG::CrossOptimizeGraph(float* data_load1, float* data_load2, const float * query, std::vector<unsigned> seeds1, std::vector<unsigned> seeds2, unsigned num, unsigned num2, unsigned query_num, size_t K, const Parameters &parameters, std::vector<std::vector<std::vector<unsigned>>> &indices, unsigned dim){
  data1_ = data_load1;
  data2_ = data_load2;

  // crossShardBound(data1_, data2_, dim, 3);
  CompactGraph shard1 = split_forest_[0];
  unsigned max_width = width_vec_[0];
  
  OptimizeGraph2(data1_, shard1, dim, max_width, num);

  CompactGraph shard2 = split_forest_[1];
  max_width = width_vec_[1];
  
  OptimizeGraph2(data2_, shard2, dim, max_width, num2);
  std::chrono::duration<double, std::milli> total_duration(0);
  for (unsigned idx = 0; idx < query_num; idx++){
    std::vector<Neighbor> retset (K);
    std::vector<Neighbor> retset2 (K);
    const float* tmp_query = query + idx * dim;
    float min_dis = FLT_MAX;
    unsigned min_seed = 0;
    std::vector<unsigned> search_entries;
    for (int i = 0; i < seeds1.size(); i++){
      unsigned id = seeds1[i];
      float dis = euclideanDistance(tmp_query, data1_ + id * dim, dim);
    }
    if (std::find(search_entries.begin(), search_entries.end(), min_seed) == search_entries.end()){
      search_entries.push_back(min_seed);
    }
    // std::vector<unsigned> cross_entries;
    // for (int j = 0; j < cross_links_[0][ep].size(); j++){
    //   unsigned id = cross_links_[0][ep][j];
    //   if(std::find(cross_entries.begin(), cross_entries.end(), id) == cross_entries.end()){
    //     cross_entries.push_back(id);
    //   }
    // }
    std::vector<unsigned> GV;
    unsigned ep = search_entries[0];
    // auto s1 = std::chrono::high_resolution_clock::now();
    retset = SearchSubGraphOpt(0, GV, num, ep, data1_, tmp_query, distance_, opt_graph_, node_size, data_len, K, parameters, dim);
    // for (unsigned i = 0; i < 1; i++) {
    //   unsigned ep = search_entries[i];
    //   std::vector<Neighbor> tmp (K);
    //   // auto s1 = std::chrono::high_resolution_clock::now();
    //   retset = SearchSubGraphOpt(0, GV, num, ep, data1_, tmp_query, distance_, opt_graph_, node_size, data_len, K, parameters, dim);
    // }
    for (unsigned j = 0; j < K; j++){
      indices[0][idx][j] = retset[j].id;
      GV.push_back(retset[j].id);
    }

    auto start2 = std::chrono::high_resolution_clock::now();
    retset2 = SearchSubGraphOpt(1, GV, num2, ep, data2_, tmp_query, distance2_, opt_graph2_, node_size2, data_len2, K, parameters, dim);
    auto end2 = std::chrono::high_resolution_clock::now();
    total_duration += end2 - start2;
    // for (unsigned i = 0; i < 1; i++) {
    //   unsigned ep = search_entries[i];
    //   std::vector<Neighbor> tmp (K);
    //   retset2 = SearchSubGraphOpt(1, GV, num2, ep, data2_, tmp_query, distance2_, opt_graph2_, node_size2, data_len2, K, parameters, dim);
    // }
    for (unsigned j = 0; j < K; j++){
      indices[1][idx][j] = retset2[j].id;
    }
  }
  std::cout << "Search Time of Shard 2: " << total_duration.count() / 1000 << " s" << std::endl;
}



// Search in each shard
std::vector<Neighbor> IndexNSG::SearchSubGraphOpt(unsigned tag, std::vector<unsigned> GV, unsigned nd, unsigned ep, const float *data, const float *query, efanna2e::Distance *dist, char* opt_graph, unsigned ns, unsigned dl, size_t K,
                                  const Parameters &parameters, unsigned dim) {
  unsigned L = parameters.Get<unsigned>("L_search");
  DistanceL2 *dist_fast = (DistanceL2 *)dist;

  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  
  boost::dynamic_bitset<> flags{nd, 0};
  unsigned tmp_l = 0;
  unsigned *neighbors = (unsigned *)(opt_graph + ns * ep + dl);
  unsigned MaxM_ep = *neighbors;
  neighbors++;
  
  if (tag == 0){
    for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
      init_ids[tmp_l] = neighbors[tmp_l];
      flags[init_ids[tmp_l]] = true;
    }
  }
  else{
    for (; tmp_l < L && tmp_l < GV.size(); tmp_l++) {
      init_ids[tmp_l] = GV[tmp_l];
      flags[init_ids[tmp_l]] = true;
    }
  }
  // for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
  //   init_ids[tmp_l] = neighbors[tmp_l];
  //   flags[init_ids[tmp_l]] = true;
  // }
  
  while (tmp_l < L) {
    unsigned id = rand() % nd;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd) continue;
    _mm_prefetch(opt_graph + ns * id, _MM_HINT_T0);
  }
  L = 0;

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd) continue;
    float *x = (float *)(opt_graph + ns * id);
    // float norm_x = *x;
    x++;
    // float dist = dist_fast->compare(x, query, (unsigned)dim);
    float dist = euclideanDistance(x, query, dim);
    // std::cout << dist << std::endl;
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
    L++;
  }
  
  // std::cout<<L<<std::endl;
  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      _mm_prefetch(opt_graph + ns * n + dl, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph + ns * n + dl);
      unsigned MaxM = *neighbors;
      neighbors++;
      for (unsigned m = 0; m < MaxM; ++m)
        _mm_prefetch(opt_graph + ns * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (flags[id]) continue;
        flags[id] = 1;
        float *data = (float *)(opt_graph + ns * id);
        // float norm = *data;
        data++;
        float dist = euclideanDistance(query, data, dim);
        // float dist = dist_fast->compare(query, data,(unsigned)dim);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        // if(L+1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  return std::vector<Neighbor>({retset.begin(), retset.begin() + K});
}



// optimize the subgraph within each shard
void IndexNSG::OptimizeGraph2(const float *data, std::vector<std::vector<unsigned>> subgraph, unsigned dimension, unsigned width, size_t total_num) {  // use after build or load
  if (data_len == 0){
    data_len = (dimension + 1) * sizeof(float);
    neighbor_len = (width + 1) * sizeof(unsigned);
    node_size = data_len + neighbor_len;
    opt_graph_ = (char *)malloc(node_size * total_num);
    DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;

    for (unsigned i = 0; i < total_num; i++) {
      char *cur_node_offset = opt_graph_ + i * node_size;
      float cur_norm = dist_fast->norm(data + i * dimension, dimension);
      std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
      std::memcpy(cur_node_offset + sizeof(float), data + i * dimension,
                  data_len - sizeof(float));

      cur_node_offset += data_len;
      unsigned k = subgraph[i].size();
      std::memcpy(cur_node_offset, &k, sizeof(unsigned));
      std::memcpy(cur_node_offset + sizeof(unsigned), subgraph[i].data(),
                  k * sizeof(unsigned));
    }
  }
  else{
    data_len2 = (dimension + 1) * sizeof(float);
    neighbor_len2 = (width + 1) * sizeof(unsigned);
    node_size2 = data_len2 + neighbor_len2;
    opt_graph2_ = (char *)malloc(node_size2 * total_num);
    DistanceFastL2 *dist_fast2 = (DistanceFastL2 *)distance2_;


    for (unsigned i = 0; i < total_num; i++) {
      char *cur_node_offset = opt_graph2_ + i * node_size2;
      float cur_norm = dist_fast2->norm(data + i * dimension, dimension);
      std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
      std::memcpy(cur_node_offset + sizeof(float), data + i * dimension,
                  data_len2 - sizeof(float));

      cur_node_offset += data_len2;
      unsigned k = subgraph[i].size();
      std::memcpy(cur_node_offset, &k, sizeof(unsigned));
      std::memcpy(cur_node_offset + sizeof(unsigned), subgraph[i].data(),
                  k * sizeof(unsigned));
    }
  }
}



void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root]) cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
    if (next == (nd_ + 1)) {
      s.pop();
      if (s.empty()) break;
      tmp = s.top();
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}


void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      // std::cout << pool[i].id << '\n';
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}


void IndexNSG::tree_grow(const Parameters &parameter, unsigned root) {
  // unsigned root = ep_;
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned unlinked_cnt = 0;
  while (unlinked_cnt < nd_) {
    DFS(flags, root, unlinked_cnt);
    std::cout << unlinked_cnt << '\n';
    if (unlinked_cnt >= nd_) break;
    findroot(flags, root, parameter);
    // std::cout << "new root"<<":"<<root << '\n';
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}


// void IndexNSG::tree_grow(const Parameters &parameter, unsigned root, CompactGraph &backup_fg) {
//   // unsigned root = ep_;
//   boost::dynamic_bitset<> flags{nd_, 0};
//   unsigned unlinked_cnt = 0;
//   while (unlinked_cnt < nd_) {
//     DFS(flags, root, unlinked_cnt);
//     std::cout << unlinked_cnt << '\n';
//     if (unlinked_cnt >= nd_) break;
//     findroot(flags, root, parameter);
//     // std::cout << "new root"<<":"<<root << '\n';
//   }
//   for (size_t i = 0; i < nd_; ++i) {
//     if (backup_fg[i].size() > width) {
//       width = backup_fg[i].size();
//     }
//   }
// }



void IndexNSG::inshardBound(std::map<unsigned, std::vector<unsigned >> &graph, std::vector<unsigned> inseed_nodes, unsigned seed, unsigned dim, int B){
  std::map<unsigned, unsigned> flag;
  for(unsigned i = 0; i < inseed_nodes.size(); i++){
    flag.insert(std::pair<unsigned, unsigned> (inseed_nodes[i], 0));
  }
  for (auto it = graph.begin(); it != graph.end(); it++){
    unsigned tmp = it->first;
    std::vector<unsigned> neighbors = it->second;
    if (flag[tmp] == 0){
      flag[tmp] = 1;
      std::vector<unsigned> diff;
      std::set_difference(neighbors.begin(), neighbors.end(), inseed_nodes.begin(), inseed_nodes.end(), std::inserter(diff, diff.begin()));
      if (diff.size() == 1) continue;
      for(unsigned i = 0; i < diff.size(); i++){
        float distance = euclideanDistance(data_ + diff[i] * dim, data_ + tmp * dim, dim);
        if (distance < B){
          if ((std::find(inseed_nodes.begin(), inseed_nodes.end(), diff[i]) != inseed_nodes.end()) && (std::find(inseed_nodes.begin(), inseed_nodes.end(), tmp) != inseed_nodes.end())){
            graph[tmp].push_back(diff[i]);
          }
        }
      }
    }
  }
}


// void IndexNSG::crossShardBound(const float* data_load1, const float* data_load2, unsigned dim, int Bound){
//   for (unsigned i = 0; i < split_forest_.size() - 1; i++){
//     std::map<unsigned, std::map<unsigned, std::vector<unsigned >>> tmp_tree = split_forest_[i];
//     std::vector<unsigned> tmpSeedSet;
//     for (auto it = tmp_tree.begin(); it != tmp_tree.end(); it++){
//       tmpSeedSet.push_back(it->first);
//     }
//     for (unsigned j = i + 1; j < split_forest_.size(); j++){
//       std::map<unsigned, std::map<unsigned, std::vector<unsigned >>> tmp_tree2 = split_forest_[j];
//       std::vector<unsigned> tmpSeedSet2;
//       for (auto it = tmp_tree2.begin(); it != tmp_tree2.end(); it++){
//         tmpSeedSet2.push_back(it->first);
//       }
//       std::vector<unsigned> crossNeighbors;
//       // std::map<unsigned, std::vector<unsigned>> submap;
//       // submap.insert(std::pair<unsigned, std::vector<unsigned >> (j, crossNeighbors));
//       // cross_links_.insert(std::pair<unsigned, std::map<unsigned, std::vector<unsigned >>> (i, submap));
//       std::map<unsigned, std::map<unsigned, unsigned>> distance_dict;
//       for (unsigned seed1 : tmpSeedSet){
//         for (unsigned seed2 : tmpSeedSet2){
//           float distance = euclideanDistance(data_load1 + seed1 * dim, data_load2 + seed2 * dim, dim);
//           distance_dict[seed1][seed2] = distance;
//           distance_dict[seed2][seed1] = distance;
//           if (distance < Bound){
//             cross_links_[i][seed1].push_back(seed2);
//             cross_links_[j][seed2].push_back(seed1);
//           }
//         }
//       }
//       for (unsigned seed1: tmpSeedSet){
//         if (cross_links_[i].count(seed1) == 0){
//           std::vector<unsigned> tmp_vector;
//           cross_links_[i].insert(std::pair<unsigned, std::vector<unsigned >> (seed1, tmp_vector));
//           // sort the distance_dict[seed1] by value
//           std::vector<std::pair<unsigned, unsigned>> distance_vector(distance_dict[seed1].begin(), distance_dict[seed1].end());
//           std::sort(distance_vector.begin(), distance_vector.end(), [](const std::pair<unsigned, unsigned> &a, const std::pair<unsigned, unsigned> &b){
//             return a.second < b.second;
//           });
//           cross_links_[i][seed1].push_back(distance_vector[0].first);
//         }
//       }
//       for (unsigned seed2: tmpSeedSet2){
//         if (cross_links_[j].count(seed2) == 0){
//           std::vector<unsigned> tmp_vector;
//           cross_links_[j].insert(std::pair<unsigned, std::vector<unsigned >> (seed2, tmp_vector));
//           std::vector<std::pair<unsigned, unsigned>> distance_vector(distance_dict[seed2].begin(), distance_dict[seed2].end());
//           std::sort(distance_vector.begin(), distance_vector.end(), [](const std::pair<unsigned, unsigned> &a, const std::pair<unsigned, unsigned> &b){
//             return a.second < b.second;
//           });
//           cross_links_[j][seed2].push_back(distance_vector[0].first);
//         }
//       }
//     }
//   }
// }



}

