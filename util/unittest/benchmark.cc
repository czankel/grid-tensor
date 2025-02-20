#include <gtest/gtest.h>


#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include <execution>

#include <grid/util/worker.h>

template <class ExecutionPolicy>
int test_stl(const std::vector<double>& X, ExecutionPolicy&& policy)
{
    std::vector<double> Y(X.size());
    const auto start = std::chrono::high_resolution_clock::now();
    std::transform(policy, X.cbegin(), X.cend(), Y.begin(), [](double x){
        volatile double y = std::sin(x);
        return y;
    });
    const auto stop = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return diff.count();
}

struct size3
{
  size_t operator[](const size_t i)
  {
    //static_assert(i < 3);
    return reinterpret_cast<size_t*>(this)[i];
  }
  size_t x;
  size_t y;
  size_t z;
};

template <typename T>
bool SinusJob(size3 position, size3 dimensions, T* d, const T* x)
{
  d[position[0]] = std::sin(x[position[0]]);
  return true;
}

int test_worker(const std::vector<double>& X)
{
  grid::Worker worker;

  std::vector<double> Y(X.size());
  const auto start = std::chrono::high_resolution_clock::now();

  // FIXME: should we convert to span/mdspan?
  const double* x = &*X.cbegin();
  double* d = &*Y.begin();

  for (size_t i = 0; i < X.size(); i++)
  {
    size3 pos{ i, 0, 0 };
    size3 dim{ X.size(), 0, 0 };
    worker.Post(SinusJob<double>, pos, dim, d, x);
  }

#if 0
  std::transform(policy, X.cbegin(), X.cend(), Y.begin(), [](double x){
      volatile double y = std::sin(x);
      return y;
  });
#endif
  const auto stop = std::chrono::high_resolution_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  return diff.count();
}

#if 0
int test_openmp(const std::vector<double>& X)
{
    std::vector<double> Y(X.size());
    const auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (size_t i = 0; i < X.size(); ++i) {
        volatile double y = std::sin(X[i]);
        Y[i] = y;
    }
    const auto stop = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return diff.count();
}
#endif

TEST(Worker, Benchmark)
{
    const size_t N = 10000000;
    std::vector<double> data(N);
    std::iota(data.begin(), data.end(), 1);
    //std::cout << "OpenMP:        " << test_openmp(data) << std::endl;
    std::cout << "Tensor Worker  " << test_worker(data) << std::endl;
    std::cout << "STL seq:       " << test_stl(data, std::execution::seq) << std::endl;
    std::cout << "STL par:       " << test_stl(data, std::execution::par) << std::endl;
    std::cout << "STL par_unseq: " << test_stl(data, std::execution::par_unseq) << std::endl;
    std::cout << "STL unseq:     " << test_stl(data, std::execution::unseq) << std::endl;
}
