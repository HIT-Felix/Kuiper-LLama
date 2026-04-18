#include "base/profiler.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ProfileStat {
  uint64_t calls = 0;
  double total_ms = 0.0;
  double min_ms = std::numeric_limits<double>::max();
  double max_ms = 0.0;
};

bool ParseEnabledEnv() {
  const char* value = std::getenv("KUIPER_PROFILE");
  if (!value) {
    return false;
  }
  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(), ::tolower);
  return text == "1" || text == "true" || text == "on" || text == "yes";
}

std::unordered_map<std::string, ProfileStat>& Stats() {
  static auto* stats = new std::unordered_map<std::string, ProfileStat>();
  return *stats;
}

std::mutex& StatsMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::atomic<bool>& ReportRegistered() {
  static auto* registered = new std::atomic<bool>(false);
  return *registered;
}

}  // namespace

namespace base {

CudaProfiler& CudaProfiler::GetInstance() {
  static auto* profiler = new CudaProfiler();
  return *profiler;
}

CudaProfiler::CudaProfiler() : enabled_(ParseEnabledEnv()) {
  bool expected = false;
  if (enabled_ && ReportRegistered().compare_exchange_strong(expected, true)) {
    std::atexit([]() { CudaProfiler::GetInstance().Report(); });
  }
}

bool CudaProfiler::enabled() const { return enabled_; }

void CudaProfiler::AddSample(const std::string& name, float elapsed_ms) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> guard(StatsMutex());
  auto& stat = Stats()[name];
  stat.calls += 1;
  stat.total_ms += elapsed_ms;
  stat.min_ms = std::min(stat.min_ms, static_cast<double>(elapsed_ms));
  stat.max_ms = std::max(stat.max_ms, static_cast<double>(elapsed_ms));
}

void CudaProfiler::Reset() {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> guard(StatsMutex());
  Stats().clear();
}

void CudaProfiler::Report() const {
  if (!enabled_) {
    return;
  }

  std::lock_guard<std::mutex> guard(StatsMutex());
  if (Stats().empty()) {
    return;
  }

  double total_ms_all = 0.0;
  for (const auto& [_, stat] : Stats()) {
    total_ms_all += stat.total_ms;
  }

  std::fprintf(stderr, "\n[Kuiper CUDA Op Profile]\n");
  std::fprintf(stderr, "%-16s %10s %10s %14s %14s %14s %14s\n", "Op", "Calls", "Percent",
               "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)");
  std::fprintf(stderr, "Percent is relative to summed profiled op time.\n");
  std::vector<std::pair<std::string, ProfileStat>> rows{Stats().begin(), Stats().end()};
  std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.second.total_ms > rhs.second.total_ms;
  });
  uint64_t total_calls = 0;
  for (const auto& [name, stat] : rows) {
    const double avg_ms = stat.calls == 0 ? 0.0 : stat.total_ms / static_cast<double>(stat.calls);
    const double percent = total_ms_all == 0.0 ? 0.0 : (stat.total_ms / total_ms_all) * 100.0;
    const double min_ms = stat.calls == 0 ? 0.0 : stat.min_ms;
    total_calls += stat.calls;
    std::fprintf(stderr, "%-16s %10llu %9.2f%% %14.4f %14.4f %14.4f %14.4f\n", name.c_str(),
                 static_cast<unsigned long long>(stat.calls), percent, stat.total_ms, avg_ms,
                 min_ms, stat.max_ms);
  }
  std::fprintf(stderr, "%-16s %10llu %9.2f%% %14.4f\n", "TOTAL",
               static_cast<unsigned long long>(total_calls), 100.0, total_ms_all);
}

ScopedCudaProfile::ScopedCudaProfile(const char* name, cudaStream_t stream)
    : ScopedCudaProfile(name ? std::string(name) : std::string(), stream) {}

ScopedCudaProfile::ScopedCudaProfile(std::string name, cudaStream_t stream)
    : name_(std::move(name)), stream_(stream) {
  if (name_.empty() || !CudaProfiler::GetInstance().enabled()) {
    return;
  }

  cudaError_t err = cudaEventCreate(&start_);
  if (err != cudaSuccess) {
    return;
  }
  err = cudaEventCreate(&stop_);
  if (err != cudaSuccess) {
    cudaEventDestroy(start_);
    start_ = nullptr;
    return;
  }
  events_created_ = true;
  err = cudaEventRecord(start_, stream_);
  if (err != cudaSuccess) {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
    start_ = nullptr;
    stop_ = nullptr;
    events_created_ = false;
    return;
  }
  active_ = true;
}

ScopedCudaProfile::~ScopedCudaProfile() {
  if (!active_) {
    return;
  }

  if (!events_created_) {
    return;
  }

  cudaError_t err = cudaEventRecord(stop_, stream_);
  if (err != cudaSuccess) {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
    return;
  }
  err = cudaEventSynchronize(stop_);
  if (err != cudaSuccess) {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
    return;
  }

  float elapsed_ms = 0.f;
  err = cudaEventElapsedTime(&elapsed_ms, start_, stop_);
  if (err == cudaSuccess) {
    CudaProfiler::GetInstance().AddSample(name_, elapsed_ms);
  }

  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

}  // namespace base
