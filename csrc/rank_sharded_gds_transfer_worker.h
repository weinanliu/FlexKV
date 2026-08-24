#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <torch/extension.h>
#include "gds/gds_manager.h"
#include "transfer_ssd.h"
#include "ce_transfer.h"

class GDSManager;

namespace flexkv {

class ThreadPool {
public:
  ThreadPool(size_t num_threads) {
    queues_.resize(num_threads);
    mtxs_ = std::vector<std::mutex>(num_threads);
    cvs_ = std::vector<std::condition_variable>(num_threads);
  
    // create the thread pool
    stop_pool_ = false;
    for (size_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this, i]() {
        cudaSetDevice(i);
        while (true) {
          Task task;
          {
            std::unique_lock<std::mutex> lk(mtxs_[i]);
            cvs_[i].wait(lk, [&] { return stop_pool_ || !queues_[i].empty(); });
            if (stop_pool_ && queues_[i].empty())
              return;
  
            task = std::move(queues_[i].front());
            queues_[i].pop();
          }
          task(); //
        }
      });
    }
  }

  ~ThreadPool() {
    stop_pool_ = true;
    for (auto &cv : cvs_)
      cv.notify_all();
    for (auto &t : threads_)
      if (t.joinable())
        t.join();
  }

  std::future<void> enqueue(size_t i, std::function<void()> task) {
    auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
    auto fut = pkg->get_future();
    {
      std::lock_guard<std::mutex> lk(mtxs_[i]);
      queues_[i].emplace([pkg] { (*pkg)(); });
    }
    cvs_[i].notify_one();
    return fut;
  }
private:
  using Task = std::function<void()>;
  std::vector<std::thread> threads_;
  std::vector<std::queue<Task>> queues_;
  std::vector<std::mutex> mtxs_;
  std::vector<std::condition_variable> cvs_;
  std::atomic<bool> stop_pool_;
};

class RankShardedGDSTransferWorker {
 public:
  RankShardedGDSTransferWorker(
    int itemsize,
    int kv_dim,
    int num_layers,
    int num_kv_heads,

    
    std::vector<int64_t> cpu_ptrs,
    size_t cpu_num_blocks,

    bool use_ce_transfer_h2d,
    bool use_ce_transfer_d2h,
    int64_t ce_segment_threshold,
    bool ce_path_opt,
    int ce_force_path,
    bool ce_enable_memcpy2d,
    int ce_gather_threads,
    bool ce_gather_nt,

    std::vector<std::string> files,
    size_t ssd_num_blocks,
    int iouring_entries,
    int iouring_flags,

    size_t external_kv_stride,
    size_t external_block_stride,
    size_t external_layer_stride,


    const std::vector<int64_t> &gpu_block_ptrs_flat,
    int num_tensors_per_gpu,
    size_t gpu_num_blocks,

    size_t gpu_kv_stride,
    size_t gpu_block_stride,
    size_t gpu_layer_stride,
    size_t gpu_chunk_size
  );
  ~RankShardedGDSTransferWorker();

  void gds_transfer(
      const torch::Tensor& gpu_block_id_tensor,
      const torch::Tensor& ssd_block_id_tensor,
      bool is_read);

  void dram_ssd_transfer(
      const torch::Tensor &cpu_block_id_tensor,
      const torch::Tensor &ssd_block_id_tensor,
      const torch::Tensor &cpu_layer_id_list,
      bool is_read,
      int num_threads_per_device,
      bool ssd_io_opt);

  void host_dev_transfer(
      torch::Tensor &cpu_block_id_tensor,
      torch::Tensor &gpu_block_id_tensor,
      bool is_read);

 private:
  template<BackendType Type> void
  host_dev_transfer_(
      int64_t *cpu_block_ids,
      int64_t *gpu_block_ids,
      size_t num_transfers,
      bool is_read);

  template<BackendType Type> void
  gds_transfer_(
     const torch::Tensor& gpu_block_id_tensor,
     const torch::Tensor& ssd_block_id_tensor,
     bool is_read);


  int itemsize;
  int kv_dim;
  int num_layers;
  int num_kv_heads;


  std::vector<int64_t> cpu_ptrs;
  size_t cpu_num_blocks;

  std::vector<std::string> files;
  size_t ssd_num_blocks;
  int iouring_entries;
  int iouring_flags;

  size_t external_kv_stride;
  size_t external_block_stride;
  size_t external_layer_stride;


  const std::vector<int64_t>& gpu_block_ptrs_flat;
  int num_tensors_per_gpu;
  size_t gpu_num_blocks;

  size_t gpu_kv_stride;
  size_t gpu_block_stride;
  size_t gpu_layer_stride;
  size_t gpu_chunk_size;

  BackendType backend_type_;
  bool is_mla_;
  size_t tp_size;

  void **gpu_blocks_;
  std::vector<GTensorHandler> gpu_tensor_handlers_;

  size_t concurrency_;
  std::unique_ptr<std::mutex[]> slot_mutexes;
  std::vector<std::unique_ptr<GDSManager>> gds_managers_;
  std::vector<void *> buffers_;
  std::vector<cudaStream_t> slot_streams;

  std::vector<std::unique_ptr<SSDIOCTX>> ssd_ioctxes_;

  std::unique_ptr<ThreadPool> thread_pool__for_ssd_cpu;
  std::vector<cudaStream_t> streams__for_host_dev_transfer;

  bool use_ce_transfer_h2d;
  bool use_ce_transfer_d2h;
  flexkv::CETransferConfig cfg;
  std::unique_ptr<ThreadPool> thread_pool__for_cpu_gpu;
};

} // namespace flexkv
