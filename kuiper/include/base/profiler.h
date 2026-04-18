#ifndef KUIPER_INCLUDE_BASE_PROFILER_H_
#define KUIPER_INCLUDE_BASE_PROFILER_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

namespace base {

class CudaProfiler {
 public:
  static CudaProfiler& GetInstance();

  bool enabled() const;

  void AddSample(const std::string& name, float elapsed_ms);

  void Reset();

  void Report() const;

 private:
  CudaProfiler();
  ~CudaProfiler() = default;

  bool enabled_ = false;
};

class ScopedCudaProfile {
 public:
  ScopedCudaProfile(const char* name, cudaStream_t stream);
  ScopedCudaProfile(std::string name, cudaStream_t stream);
  ~ScopedCudaProfile();

 private:
  std::string name_;
  bool active_ = false;
  bool events_created_ = false;
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
  cudaStream_t stream_{};
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_PROFILER_H_
