#include "rank_sharded_gds_transfer_worker.h"

#include "logging.h"
#include "gds/layout_transform.cuh"
#include "transfer_ssd.h"
#include "transfer.cuh"

#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cufile.h>

namespace flexkv {
RankShardedGDSTransferWorker::RankShardedGDSTransferWorker(
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
  ):
    itemsize(itemsize),
    kv_dim(kv_dim),
    num_layers(num_layers),
    num_kv_heads(num_kv_heads),
    cpu_ptrs(cpu_ptrs),
    cpu_num_blocks(cpu_num_blocks),
    use_ce_transfer_h2d(use_ce_transfer_h2d),
    use_ce_transfer_d2h(use_ce_transfer_d2h),
    files(files),
    ssd_num_blocks(ssd_num_blocks),
    iouring_entries(iouring_entries),
    iouring_flags(iouring_flags),
    external_kv_stride(external_kv_stride),
    external_block_stride(external_block_stride),
    external_layer_stride(external_layer_stride),
    gpu_block_ptrs_flat(gpu_block_ptrs_flat),
    num_tensors_per_gpu(num_tensors_per_gpu),
    gpu_num_blocks(gpu_num_blocks),
    gpu_kv_stride(gpu_kv_stride),
    gpu_block_stride(gpu_block_stride),
    gpu_layer_stride(gpu_layer_stride),
    gpu_chunk_size(gpu_chunk_size)
    {



  if (num_tensors_per_gpu == 0) {

  } else if (num_tensors_per_gpu == 1) {
    this->backend_type_ = flexkv::BackendType::TRTLLM;
  } else if (num_tensors_per_gpu == num_layers) {
    this->backend_type_ = flexkv::BackendType::VLLM;
  } else if (num_tensors_per_gpu == 2 * num_layers) {
    this->backend_type_ = flexkv::BackendType::SGLANG;
  } else {
    throw std::runtime_error("Unsupported GPU block type: " + std::to_string(num_tensors_per_gpu));
  }

  if (kv_dim == 1)
    this->is_mla_ = true;
  else
    this->is_mla_ = false;

  if (0 < cpu_ptrs.size())
    this->tp_size = cpu_ptrs.size();
  else if (0 < files.size())
    this->tp_size = files.size();
  else
    throw std::runtime_error("Failed");

  FLEXKV_LOG_INFO(
    "RankShardedGDSTransferWorker constructor scalar params: "
    "num_layers=%d "
    "external_layer_stride=%lld "
    "external_kv_stride=%lld "
    "external_block_stride=%lld "
    "kv_dim=%lld "
    "itemsize=%d "
    "tp_size=%d ",
    (int)num_layers,
    static_cast<long long>(external_layer_stride),
    static_cast<long long>(external_kv_stride),
    static_cast<long long>(external_block_stride),
    static_cast<long long>(kv_dim),
    (int)itemsize,
    (int)this->tp_size
  );



  static int gc_inited = 0;
  static std::vector<cudaExecutionContext_t> kv_gcs;

  if (0 < gpu_block_ptrs_flat.size()) {
    FLEXKV_LOG_INFO(
        "RankShardedGDSTransferWorker constructor gpu_block_ptrs_flat: "
        "ptr_count=%zu",
        gpu_block_ptrs_flat.size());

    cudaError_t malloc_err = cudaMallocHost(
        (void **)&gpu_blocks_, this->tp_size * num_tensors_per_gpu * sizeof(void *));
    if (malloc_err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMallocHost failed: ") +
                               cudaGetErrorString(malloc_err));
    }
    for (size_t i = 0; i < gpu_block_ptrs_flat.size(); ++i) {
      gpu_blocks_[i] = reinterpret_cast<void *>(gpu_block_ptrs_flat[i]);
    }

    for (size_t rank = 0; rank < this->tp_size; rank++) {
      int64_t **gpu_blocks_ptr =
          reinterpret_cast<int64_t **>(gpu_blocks_ + rank * num_tensors_per_gpu);
      this->gpu_tensor_handlers_.emplace_back(
        this->backend_type_,
        gpu_blocks_ptr,
        num_layers,
        gpu_kv_stride * itemsize,
        gpu_block_stride * itemsize,
        gpu_layer_stride * itemsize);
    }

    if (gc_inited == 0) {
      gc_inited = 1;
      // for (size_t rank = 0; rank < this->tp_size; rank++) {
      //   cudaError_t err;
      //   cudaSetDevice(rank);

      //   cudaDevResource initial_SM;
      //   err = cudaDeviceGetDevResource(rank, &initial_SM, cudaDevResourceTypeSm);
      //   if (err != cudaSuccess) {
      //     throw std::runtime_error("error occurred while constructing green context");
      //   }

      //   cudaDevResource kv_res, remaining_res;
      //   unsigned int nbGroups = 1;
      //   err = cudaDevSmResourceSplitByCount(&kv_res, &nbGroups, &initial_SM, &remaining_res, 0, 8);
      //   if (err != cudaSuccess) {
      //     throw std::runtime_error("error occurred while constructing green context");
      //   }

      //   cudaDevResourceDesc_t desc;
      //   err = cudaDevResourceGenerateDesc(&desc, &kv_res, 1);
      //   if (err != cudaSuccess) {
      //     throw std::runtime_error("error occurred while generating resource description");
      //   }

      //   cudaExecutionContext_t gc;
      //   err = cudaGreenCtxCreate(&gc, desc, 0, 0);
      //   if (err != cudaSuccess) {
      //     throw std::runtime_error("error occurred while creating green context");
      //   }
      //   kv_gcs.push_back(gc);
      // }
    } else {
      // if (kv_gcs.size() != this->tp_size) {
      //   throw std::runtime_error("Mismatch in kv_gcs size and tp_size");
      // }
    }
  }

  if (0 < files.size() && 0 < gpu_block_ptrs_flat.size()) {
    FLEXKV_LOG_INFO(
        "RankShardedGDSTransferWorker Init for D2DISK and DISK2D");
    this->concurrency_ = 16;
    this->slot_mutexes = std::make_unique<std::mutex[]>(this->tp_size * this->concurrency_);
    for (size_t rank_id = 0; rank_id < this->tp_size; rank_id++) {
      cudaError_t cuda_status = cudaSetDevice(rank_id);
      if (cuda_status != cudaSuccess) {
        throw std::runtime_error(
            "cudaSetDevice failed for rank " + std::to_string(rank_id) + ": " +
            cudaGetErrorString(cuda_status));
      }
  
      std::map<int, std::vector<std::string>> ssd_files;
      ssd_files[0].push_back(files[rank_id]);
      auto gds_manager = std::make_unique<GDSManager>(ssd_files, 1, 1);
      if (!gds_manager->is_ready()) {
        throw std::runtime_error(
            "Failed to initialize GDS Manager for rank " + std::to_string(rank_id) +
            ": " + gds_manager->get_last_error());
      }
      this->gds_managers_.push_back(std::move(gds_manager));
      FLEXKV_LOG_INFO(
          "GDSManager initialized and ready: rank_id=%zu file_path=%s ready=true",
          rank_id, this->files[rank_id].c_str());


      void *buffer;
      cuda_status = cudaMalloc(&buffer, this->concurrency_ * external_block_stride * itemsize);
      if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU buffer: " + 
                                std::string(cudaGetErrorString(cuda_status)));
      }
  
      CUfileError_t status = cuFileBufRegister(buffer,
                                               this->concurrency_ * external_block_stride * itemsize, 0);
      if (status.err != CU_FILE_SUCCESS) {
  	  	std::cerr << "Buffer register failed: ";
        throw std::runtime_error("set device fail");
  	  }
      this->buffers_.push_back(buffer);
  
      for (size_t i = 0; i < this->concurrency_; i++) {
        cudaError_t err;
        cudaStream_t stream;
        // err = cudaExecutionCtxStreamCreate(&stream, kv_gcs[rank_id], cudaStreamDefault, -1);
        // if (err != cudaSuccess) {
        //   throw std::runtime_error("Failed to create GPU stream: " + 
        //                           std::string(cudaGetErrorString(err)));
        // }
        cudaStreamCreate(&stream);
        this->slot_streams.push_back(stream);
      }
    }
    FLEXKV_LOG_INFO(
        "RankShardedGDSTransferWorker %d concurrency slots per rank, "
        "%d slot streams total ",
        (int)this->concurrency_,
        (int)this->slot_streams.size());
  }

  if (0 < files.size() && 0 < cpu_ptrs.size()) {
    FLEXKV_LOG_INFO(
        "RankShardedGDSTransferWorker Init for DISK2H and H2DISK");
    this->thread_pool__for_ssd_cpu = std::make_unique<ThreadPool>(this->tp_size);
    for (size_t rank_id = 0; rank_id < this->tp_size; rank_id++) {
      std::map<int, std::vector<std::string>> ssd_files;
      ssd_files[0].push_back(files[rank_id]);
      auto ioctx = std::make_unique<SSDIOCTX>(ssd_files, 1, iouring_entries, iouring_flags);
      this->ssd_ioctxes_.push_back(std::move(ioctx));
    }
  }

  if (0 < gpu_block_ptrs_flat.size() && 0 < cpu_ptrs.size()) {
    FLEXKV_LOG_INFO(
        "RankShardedGDSTransferWorker Init for D2H and H2D");
    this->thread_pool__for_cpu_gpu = std::make_unique<ThreadPool>(this->tp_size);
    for (size_t rank = 0; rank < this->tp_size; rank++) {
      cudaSetDevice(rank);
      cudaError_t err;
      cudaStream_t stream;
      // err = cudaExecutionCtxStreamCreate(&stream, kv_gcs[rank], cudaStreamDefault, -1);
      // if (err != cudaSuccess) {
      //   throw std::runtime_error("Failed to create GPU stream: " + 
      //                           std::string(cudaGetErrorString(err)));
      // }
      cudaStreamCreate(&stream);
      this->streams__for_host_dev_transfer.push_back(stream);
    }

    this->cfg.segment_threshold = ce_segment_threshold;
    this->cfg.path_opt_enabled = ce_path_opt;
    this->cfg.force_path = ce_force_path;
    this->cfg.enable_memcpy2d = ce_enable_memcpy2d;
    this->cfg.is_blockfirst = true;
    this->cfg.num_kv_heads = num_kv_heads;
    this->cfg.gather_threads = ce_gather_threads;
    this->cfg.gather_nt = ce_gather_nt;
  }
}

RankShardedGDSTransferWorker::~RankShardedGDSTransferWorker() {
  if (0 < this->gpu_block_ptrs_flat.size()) {
    cudaFreeHost(this->gpu_blocks_);
  }
  if (0 < this->files.size() && 0 < this->gpu_block_ptrs_flat.size()) {
    for (auto &buffer : this->buffers_) {
      cuFileBufDeregister(buffer);
      cudaFree(buffer);
    }
  }
}

template<BackendType Type> void
RankShardedGDSTransferWorker::gds_transfer_(
  const torch::Tensor& gpu_block_id_tensor,
  const torch::Tensor& ssd_block_id_tensor,
  bool is_read) {
  const size_t num_transfers = gpu_block_id_tensor.size(0);
  assert (ssd_block_id_tensor.size(0) == num_transfers);
  if (num_transfers == 0) {
    return;
  }

  const int64_t *gpu_block_ids = gpu_block_id_tensor.data_ptr<int64_t>();
  const int64_t *ssd_block_ids = ssd_block_id_tensor.data_ptr<int64_t>();


  std::vector<int64_t *> d_gpu_block_ids__per_rank;
  d_gpu_block_ids__per_rank.reserve(this->tp_size);
  for (size_t rank = 0; rank < this->tp_size; rank++) {
    cudaSetDevice(rank);
    int64_t *d_gpu_block_ids;
    cudaError_t malloc_status = cudaMalloc(&d_gpu_block_ids, this->concurrency_ * sizeof(int64_t));
    if (malloc_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate d_gpu_block_ids" + std::to_string(this->concurrency_ * sizeof(int64_t)));
    }
    d_gpu_block_ids__per_rank.push_back(d_gpu_block_ids);
  }
    

  std::atomic<bool> failed{false};
  std::string error_msg;
  std::vector<std::future<void>> futures;
  futures.reserve(this->tp_size * num_transfers);

  for (size_t transfer_idx = 0; transfer_idx < num_transfers; ++transfer_idx) {
    size_t slot_id = transfer_idx % this->concurrency_;

    int64_t gpu_block_id = gpu_block_ids[transfer_idx];
    int64_t ssd_block_id = ssd_block_ids[transfer_idx];
    assert (0 <= gpu_block_id && gpu_block_id < this->gpu_num_blocks_);
    assert (0 <= ssd_block_id && ssd_block_id < this->ssd_num_blocks_);
    for (size_t rank = 0; rank < this->tp_size; rank++) {
      GDSManager& gds_manager = *(this->gds_managers_[rank]);
      futures.emplace_back(gds_manager.enqueue_task([&, rank, slot_id, gpu_block_id, ssd_block_id]() {
        cudaSetDevice(rank);

        GTensorHandler &gpu_tensor_handler = this->gpu_tensor_handlers_[rank];

        std::lock_guard<std::mutex> slot_lock(this->slot_mutexes[rank * this->concurrency_ + slot_id]);
        cudaStream_t slot_stream = slot_streams[rank * this->concurrency_ + slot_id];

        cudaStreamSynchronize(slot_stream);
        int64_t *d_my_block_id = d_gpu_block_ids__per_rank[rank] + slot_id;
        const int64_t gpu_block_id_64 = gpu_block_id;
        cudaMemcpyAsync(d_my_block_id, &gpu_block_id_64, sizeof(int64_t),
                        cudaMemcpyHostToDevice, slot_stream);

        uint64_t buffer_base = (uint64_t)(this->buffers_[rank]);
        void *gpu_buffer = (void *)(buffer_base + slot_id * this->external_block_stride * this->itemsize);

        const std::string &filename = this->files[rank];
        size_t ssd_ofst = ssd_block_id * this->external_block_stride * this->itemsize;

        cudaStreamSynchronize(slot_stream);

        if (is_read) {
          gds_manager.read(filename.c_str(), gpu_buffer, this->external_block_stride * this->itemsize, ssd_ofst);
          launch_layout_transform_kernel<Type>(
            (int64_t *)gpu_buffer,
            (this->external_layer_stride * this->itemsize) / sizeof(int64_t),
            (this->external_kv_stride * this->itemsize) / sizeof(int64_t),
            (this->external_block_stride * this->itemsize) / sizeof(int64_t),
            (this->external_kv_stride * this->itemsize) / sizeof(int64_t),
            gpu_tensor_handler,
            d_my_block_id,
            1,  // num_blocks
            this->num_layers,
            this->is_mla_,
            true,
            slot_stream
          );
        } else {
          launch_layout_transform_kernel<Type>(
            (int64_t *)gpu_buffer,
            (this->external_layer_stride * this->itemsize) / sizeof(int64_t),
            (this->external_kv_stride * this->itemsize) / sizeof(int64_t),
            (this->external_block_stride * this->itemsize) / sizeof(int64_t),
            (this->external_kv_stride * this->itemsize) / sizeof(int64_t),
            gpu_tensor_handler,
            d_my_block_id,
            1,  // num_blocks
            this->num_layers,
            this->is_mla_,
            false,
            slot_stream
          );
          cudaStreamSynchronize(slot_stream);
          gds_manager.write(filename.c_str(), gpu_buffer, this->external_block_stride * this->itemsize, ssd_ofst);
        }

      }));
    }
  }

  for (auto &f : futures) {
    f.get();
  }

  for (size_t rank = 0; rank < this->files.size(); rank++) {
    cudaSetDevice(rank);
    cudaFree(d_gpu_block_ids__per_rank[rank]);
  }
}


void RankShardedGDSTransferWorker::gds_transfer(
    const torch::Tensor& gpu_block_id_tensor,
    const torch::Tensor& ssd_block_id_tensor,
    bool is_read) {

  assert (gpu_block_id_tensor.scalar_type() == torch::kInt64
    && ssd_block_id_tensor.scalar_type() == torch::kInt64);
  assert (gpu_block_id_tensor.dim() == 1
    && ssd_block_id_tensor.dim() != 1);

  if (this->gds_managers_.size() == 0 || this->gpu_tensor_handlers_.size() == 0) {
    throw std::runtime_error("Unsupport");
    return;
  }

  switch(this->backend_type_) {
    case flexkv::BackendType::TRTLLM:
      this->gds_transfer_<flexkv::BackendType::TRTLLM>(gpu_block_id_tensor, ssd_block_id_tensor, is_read);
      break;
    case flexkv::BackendType::VLLM:
      this->gds_transfer_<flexkv::BackendType::VLLM>(gpu_block_id_tensor, ssd_block_id_tensor, is_read);
      break;
    case flexkv::BackendType::SGLANG:
      this->gds_transfer_<flexkv::BackendType::SGLANG>(gpu_block_id_tensor, ssd_block_id_tensor, is_read);
      break;
    default:
      throw std::runtime_error("Wrong!");
  }
}

void RankShardedGDSTransferWorker::dram_ssd_transfer(
    const torch::Tensor &cpu_block_id_tensor,
    const torch::Tensor &ssd_block_id_tensor,
    const torch::Tensor &cpu_layer_id_list,
    bool is_read,
    int num_threads_per_device,
    bool ssd_io_opt) {

  if (this->ssd_ioctxes_.size() == 0 || this->cpu_ptrs.size() == 0) {
    throw std::runtime_error("Unsupport");
    return;
  }

  std::vector<std::future<void>> futures;
  futures.reserve(this->tp_size);
  for (size_t rank = 0; rank < this->tp_size; rank++) {
    futures.emplace_back(this->thread_pool__for_ssd_cpu->enqueue(rank, [&, rank]() {
      transfer_kv_blocks_ssd(
        *(this->ssd_ioctxes_[rank]),
        cpu_layer_id_list, // cpu_layer_id_list
        this->cpu_ptrs[rank], // cpu_tensor_ptr
        ssd_block_id_tensor,
        cpu_block_id_tensor,
        this->external_layer_stride * this->itemsize, // cpu_layer_stride_in_bytes
        this->external_kv_stride * this->itemsize, // cpu_kv_stride_in_bytes
        this->external_layer_stride * this->itemsize, // ssd_layer_stride_in_bytes
        this->external_kv_stride * this->itemsize, // ssd_kv_stride_in_bytes
        this->external_kv_stride * this->itemsize, // chunk_size_in_bytes
        this->external_block_stride * this->itemsize,
        is_read,
        this->ssd_num_blocks,
        1,
        num_threads_per_device,
        this->is_mla_,
        ssd_io_opt
      );
    }));
  }
  for (auto &f : futures) {
    f.get();
  }
}


template<BackendType Type> void
RankShardedGDSTransferWorker::host_dev_transfer_(
      int64_t *cpu_block_ids,
      int64_t *gpu_block_ids,
      size_t num_transfers,
      bool is_read) {
  std::vector<std::future<void>> futures;
  futures.reserve(this->tp_size);
  bool use_ce;
  if (is_read) {
    use_ce = this->use_ce_transfer_h2d;
  } else {
    use_ce = this->use_ce_transfer_d2h;
  }
  for (size_t rank = 0; rank < this->tp_size; rank++) {
    futures.emplace_back(this->thread_pool__for_cpu_gpu->enqueue(rank, [&, rank]() {
      transfer_kv_blocks<Type>(num_transfers, 0, this->num_layers,
                               gpu_block_ids, this->gpu_tensor_handlers_[rank], 0,
                               cpu_block_ids, (void *)this->cpu_ptrs[rank],
                               this->external_kv_stride * this->itemsize,
                               this->external_layer_stride * this->itemsize,
                               this->external_block_stride * this->itemsize,
                               0, this->external_kv_stride * this->itemsize,
                               this->streams__for_host_dev_transfer[rank],
                               4, is_read,
                               use_ce, this->kv_dim,
                               this->gpu_block_stride * this->itemsize,
                               true, this->cfg, false);
    }));
  }
  for (auto &f : futures) {
    f.get();
  }
}

void RankShardedGDSTransferWorker::host_dev_transfer(
      torch::Tensor &cpu_block_id_tensor,
      torch::Tensor &gpu_block_id_tensor,
      bool is_read) {

  assert (cpu_block_id_tensor.scalar_type() == torch::kInt64
    && gpu_block_id_tensor.scalar_type() == torch::kInt64);
  assert (cpu_block_id_tensor.dim() == 1
    && gpu_block_id_tensor.dim() != 1);


  const size_t num_transfers = cpu_block_id_tensor.size(0);
  assert (gpu_block_id_tensor.size(0) == num_transfers);
  if (num_transfers == 0) {
    return;
  }

  if (this->cpu_ptrs.size() == 0 || this->gpu_tensor_handlers_.size() == 0) {
    throw std::runtime_error("Unsupport");
    return;
  }

  int64_t *cpu_block_ids = cpu_block_id_tensor.data_ptr<int64_t>();
  int64_t *gpu_block_ids = gpu_block_id_tensor.data_ptr<int64_t>();

  switch(this->backend_type_) {
    case flexkv::BackendType::TRTLLM:
      this->host_dev_transfer_<flexkv::BackendType::TRTLLM>(cpu_block_ids, gpu_block_ids, num_transfers, is_read);
      break;
    case flexkv::BackendType::VLLM:
      this->host_dev_transfer_<flexkv::BackendType::VLLM>(cpu_block_ids, gpu_block_ids, num_transfers, is_read);
      break;
    case flexkv::BackendType::SGLANG:
      this->host_dev_transfer_<flexkv::BackendType::SGLANG>(cpu_block_ids, gpu_block_ids, num_transfers, is_read);
      break;
    default:
      throw std::runtime_error("Wrong!");
  }
}
} // namespace flexkv
