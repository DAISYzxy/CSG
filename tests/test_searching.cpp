#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>
#include <eigen-3.4.0/Eigen/Dense>
#include <eigen-3.4.0/Eigen/Eigenvalues>
#include <numeric> // for std::accumulate
#include <iomanip> // for std::setprecision

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  // std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

bool loadFromIvecs(const std::string& filename, std::vector<unsigned>& seed_points) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Could not open file for reading.\n";
        return false;
    }

    // Read the size of the vector
    unsigned size = 0;
    infile.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (infile.fail()) {
        std::cerr << "Error: Could not read size from file.\n";
        return false;
    }

    // Resize the vector to hold all elements
    seed_points.resize(size);

    // Read the vector elements
    infile.read(reinterpret_cast<char*>(seed_points.data()), sizeof(unsigned) * size);
    if (infile.fail()) {
        std::cerr << "Error: Could not read elements from file.\n";
        return false;
    }

    infile.close();
    return true;
}


void Load_nn_graph(const char *filename, std::vector<std::vector<unsigned> > &nn_graph) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  nn_graph.resize(num);
  nn_graph.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    nn_graph[i].resize(k);
    nn_graph[i].reserve(kk);
    in.read((char *)nn_graph[i].data(), k * sizeof(unsigned));
  }
  in.close();
}


void save_result(const char* filename,
                 std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}


void Print_nn_graph(std::vector<std::vector<unsigned> > nn_graph) {
  for (size_t i = 0; i < nn_graph.size(); ++i) {
    std::cout << "Node " << i << ": ";
    for (size_t j = 0; j < nn_graph[i].size(); ++j) {
      std::cout << nn_graph[i][j];
      if (j < nn_graph[i].size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
}


float precision_K(int N, std::vector<std::vector<unsigned>> &ground_truth, std::vector<std::vector<unsigned >> &result) {
  std::vector<float> precision;
  size_t len = result.size();
  for (size_t i = 1; i < len; i++) {
    int correct = 0;
    std::vector<unsigned> gt = std::vector<unsigned>({ground_truth[i].begin(), ground_truth[i].begin()+N});
    std::vector<unsigned> res = std::vector<unsigned>({result[i].begin(), result[i].begin() + N});
    for (int j = 0; j < N; j++){
      if (std::find(gt.begin(), gt.end(), res[j]) != gt.end()) {
        correct++;
      }
    }
    float pre = (float)correct / N;
    precision.push_back(pre);
  }
  float sum = std::accumulate(precision.begin(), precision.end(), 0.0f);
  float average = sum / precision.size();
  float average_percentage = average * 100.0f;
  return average_percentage;
}



int main(int argc, char** argv) {
  if (argc != 12) {
    std::cout << argv[0]
              << " data_file data_file2 query_file index_file1 index_file2 seeds_file1 seeds_file2 ground_truth1 ground_truth2 L K"
              << std::endl;
    exit(-1);
  }

  float* data_load = NULL;
  float* data_load2 = NULL;
  float* query = NULL;
  unsigned points_num, dim;
  unsigned points_num2, dim2;
  unsigned query_num, query_dim;
  load_data(argv[1], data_load, points_num, dim);
  load_data(argv[2], data_load2, points_num2, dim2);
  load_data(argv[3], query, query_num, query_dim);

  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
  index.Load(argv[4]);
  index.Load(argv[5]);

  std::vector<unsigned> seed_points1;
  if (loadFromIvecs(argv[6], seed_points1)) {
      std::cout << "Seed points loaded successfully. Contents:\n";
      for (unsigned value : seed_points1) {
          std::cout << value << ' ';
      }
      std::cout << '\n';
  } else {
      std::cerr << "Failed to load seed points from file.\n";
  }

  std::vector<unsigned> seed_points2;
  if (loadFromIvecs(argv[7], seed_points2)) {
      std::cout << "Seed points loaded successfully. Contents:\n";
      for (unsigned value : seed_points2) {
          std::cout << value << ' ';
      }
      std::cout << '\n';
  } else {
      std::cerr << "Failed to load seed points from file.\n";
  }

  std::vector<std::vector<unsigned > > ground_truth1;
  Load_nn_graph(argv[8], ground_truth1);
  std::vector<std::vector<unsigned > > ground_truth2;
  Load_nn_graph(argv[9], ground_truth2);

  unsigned L = (unsigned)atoi(argv[10]);
  unsigned K = (unsigned)atoi(argv[11]);

  std::vector<std::vector<std::vector<unsigned>>> res(2);
  for (unsigned i = 0; i < 2; i++) res[i].resize(query_num);
  for (unsigned i = 0; i < query_num; i++) res[0][i].resize(K);
  for (unsigned i = 0; i < query_num; i++) res[1][i].resize(K);
  
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);
  
  index.CrossOptimizeGraph(data_load, data_load2, query, seed_points1, seed_points2, points_num, points_num2, query_num, K, paras, res, dim);
  // for (unsigned i = 0; i < 3; i++){
  //   std::cout << "query " << i << " shard 1: ";
  //   for (unsigned j = 0; j < K; j++){
  //     std::cout << res[0][i][j] << " ";
  //   }
  //   std::cout << "\n";
  //   std::cout << "query " << i << " shard 2: ";
  //   for (unsigned j = 0; j < K; j++){
  //     std::cout << res[1][i][j] << " ";
  //   }
  //   std::cout << "\n";
  // }
  float average_percentage1 = precision_K(K, ground_truth1, res[0]);
  float average_percentage2 = precision_K(K, ground_truth2, res[1]);

  std::cout << std::fixed << std::setprecision(1);
  std::cout << "average percentage of shard 1: " << average_percentage1 << "\n";
  std::cout << "average percentage of shard 2: " << average_percentage2 << "\n";

  return 0;
}
