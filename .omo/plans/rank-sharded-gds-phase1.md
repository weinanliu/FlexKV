# rank-sharded-gds-phase1 - Work Plan

## TL;DR (For humans)

**What you'll get:** A Phase1 implementation that adds a reusable rank-sharded GDS transfer worker in `flexkv/transfer/worker.py` and a benchmark-only rank-sharded GPU-SSD worker factory in `benchmarks/benchmark_workers.py`. Existing `SSDAllocator`, `StorageEngine`, `TransferEngine`, `GDSTransferWorker`, and `tpGDSTransferWorker` remain unchanged, and no dedicated rank-sharded allocator is added to `flexkv/storage/allocator.py`.

**Why this approach:** The user clarified that rank-sharded SSD placement should be constructed in the benchmark helper using regular `SSDAllocator` instances, one per TP rank, instead of adding a production allocator. Rank IDs are derived from `tp_size` as `range(tp_size)`, not passed explicitly.

**What it will NOT do:** It will not add a new rank-sharded allocator class; it will not replace existing `SSDAllocator`; it will not touch `StorageEngine` or `TransferEngine`; it will not wire a new production worker map; it will not add C++ code; it will not add topology-aware NUMA/PCIe placement.

**Effort:** Medium-High
**Risk:** Medium - the main risk is adapting symmetric rank-sharded files to the existing `c_ext.GDSManager` read/write API without introducing a misleading local/global block model.
**Decisions to sanity-check:** No new allocator; rank-sharded SSD placement lives in `benchmarks/benchmark_workers.py` using regular `SSDAllocator` instances; new worker in `flexkv/transfer/worker.py`; no explicit `tp_rank`; block IDs are rank-file block IDs; serial per-rank calls to existing `c_ext.GDSManager`; no `StorageEngine` / `TransferEngine` wiring in Phase1.

Your next move: approve the plan, then run a high-accuracy review before execution. Full execution detail follows below.

---

> TL;DR (machine): <1 line - effort, risk, deliverables>

## Scope
### Must have
- Add a new reusable rank-sharded GDS transfer worker in `flexkv/transfer/worker.py`.
- Add benchmark-only rank-sharded GPU-SSD placement in `benchmarks/benchmark_workers.py` using one regular `SSDAllocator` per TP rank.
- Existing `SSDAllocator`, `StorageEngine`, `TransferEngine`, `GDSTransferWorker`, and `tpGDSTransferWorker` remain unchanged.
- No explicit `tp_rank` argument; derive rank IDs from `range(tp_size)`.
- Serial per-rank calls to existing `c_ext.GDSManager` only; `transfer_kv_blocks_gds` remains unused by the rank-sharded path.
- No new C++ code and no `bindings.cpp` changes.
- Add focused verification that exercises allocation mapping and worker call sequence without requiring real GDS hardware unless available.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No production `StorageEngine` changes.
- No production `TransferEngine` changes.
- No replacement or semantic change to existing `SSDAllocator`.
- No new C++ code or `bindings.cpp` changes.
- No new production worker map or `TransferType` wiring.
- No explicit `tp_rank` parameter in the Phase1 API.
- No NUMA/PCIe topology-aware placement in Phase1.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after with focused unit tests for the new worker and benchmark helper behavior.
- Evidence: `.omo/evidence/task-* -rank-sharded-gds-phase1.<ext>`.
- Required checks: Python syntax/import, benchmark helper placement assertions, worker transfer mapping assertions, optional GDS benchmark smoke when `transfer_kv_blocks_gds` is compiled.

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | None | 2, 3, 4 | None |
| 2 | 1 | 3, 6 | None |
| 3 | 1, 2 | 4, 5, 6 | None |
| 4 | 1, 2, 3 | 5, 6 | None |
| 5 | 4 | 7 | None |
| 6 | 1-5 | Final verification | None |
| 7 | 5, 6 | Final verification | None |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Remove dedicated rank-sharded allocator and use regular SSD allocator per rank
  What to do / Must NOT do: Do not add a new rank-sharded allocator class; use one regular `SSDAllocator` per TP rank in the benchmark helper, derive a per-rank SSD `KVCacheLayout` with `div_head(tp_size)`, and keep rank-sharded metadata construction outside `allocator.py`. Each rank file stores all block ids from the original layout: `num_blocks_per_file = layout.num_block`. Keep worker metadata minimal: `tp_rank__to__file_path` maps `rank_id -> file_path`, and the worker derives `tp_group_size` from that map. Do not add `per_rank_blocks` or `blocks_per_rank`; do not divide `layout.num_block` by `tp_size` for block count; do not replace or change `SSDAllocator`; do not touch `StorageEngine`.
  Parallelization: Wave 1 | Blocked by: None | Blocks: rank block id semantics and rank-sharded worker.
  References (executor has NO interview context - be exhaustive): `flexkv/storage/allocator.py:592-664`; `flexkv/storage/allocator.py:675-758`; `flexkv/common/storage.py:294-344`; `benchmarks/benchmark_workers.py:229-323`.
  Acceptance criteria (agent-executable): Run a Python snippet importing `benchmarks.benchmark_workers` and monkeypatching `GPUAllocator.allocate` / `SSDAllocator.allocate`; call `create_rank_sharded_gpu_ssd_worker(...)` with a temporary cache dir, `tp_size=8`, `num_devices=4`, `num_ssd_blocks=32`, and `dtype=torch.bfloat16`; assert 8 GPU allocations, 8 SSD allocations, 8 files exist, each device/cache-dir entry has 2 files, all files have identical expected size, `num_blocks_per_file == 32`, file size equals `layout.get_total_elements() * dtype.itemsize // tp_size`, and `ssd_layout.get_chunk_size() == layout.tokens_per_block * (layout.num_head // tp_size) * layout.head_size * dtype.itemsize` when the provided layout represents the TP-rank GPU KV layout.
  QA scenarios (name the exact tool + invocation): happy + failure, Evidence `.omo/evidence/task-1-rank-sharded-gds-phase1.md`. Happy: helper returns expected per-rank file map. Failure: empty `cache_dir` or invalid `file_prefix` raises a clear error; current design does not add a dedicated allocator.
  Commit: Y | `feat(bench): use regular SSD allocator per rank`

- [x] 2. Define rank block id semantics without local/global mapping helpers
  What to do / Must NOT do: Do not add any rank-local-to-global block mapping helper. The benchmark helper should expose only `tp_rank__to__file_path` mapping rank id to the selected rank file path; a `block_id` means the block offset inside the selected rank file, and the same `block_id` exists symmetrically in every rank file. Define chunk offset semantics for Task 3 worker use: `chunk_offset = layer_id * layer_stride + is_v * kv_stride + blk_stride * blk_id`; chunk is the minimum transfer unit. If an internal GDSManager compatibility adapter is needed later, name it `gds_block_ids` and document it as an implementation detail, not as a user/worker block concept.
  Parallelization: Wave 1 | Blocked by: Todo 1 | Blocks: rank-sharded GDS worker correctness.
  References: `csrc/gds/gds_manager.cpp:577-593`; `csrc/gds/gds_manager.cpp:689-711`; `csrc/bindings.cpp:725-739`; `csrc/gds/gds_manager.h:322-342`.
  Acceptance criteria (agent-executable): Static/import-level checks assert that benchmark helper metadata contains one symmetric file per rank, `tp_rank__to__file_path` maps rank id to file path, and no allocator API contains `local_block` / `global_block` naming or rank-local-to-global conversion helpers. Unit assertions verify that for `tp_size=8`, `num_devices=4`, `num_blocks_per_file=32`, every rank has block offsets `[0,1,2,...,num_blocks_per_file-1]` in its own file; no rank-to-global mapping assertion remains.
  QA scenarios: happy + failure, Evidence `.omo/evidence/task-2-rank-sharded-gds-phase1.md`. Happy: helper metadata and naming match rank-file block semantics. Failure: any helper or plan text still describes rank files using local/global block mapping.
  Commit: Y | `feat(bench): define rank-file block semantics`

- [x] 3. Add `RankShardedGDSTransferWorker` in `flexkv/transfer/worker.py`
  What to do / Must NOT do: Add a new reusable worker class in `worker.py` with the simplified API:
  ```python
  def __init__(
      self,
      worker_id: int,
      transfer_conn: Optional[Connection],
      finished_ops_queue: MPQueue,
      gpu_blocks: List[List[TensorSharedHandle]],
      gpu_kv_layouts: List[KVCacheLayout],
      dtype: torch.dtype,
      ssd_layout: KVCacheLayout,
      tp_rank__to__file_path: Dict[int, str],
  ) -> None:
  ```
  The constructor must call `super().__init__(worker_id, transfer_conn, finished_ops_queue, None)` and derive `self.tp_group_size = len(tp_rank__to__file_path)`; it must not accept `op_buffer_tensor`, `ssd_files`, `tp_group_size`, `ssd_kv_layout`, explicit `num_blocks_per_file`, `rank_sharded_metadata`, `gpu_device_id`, or `gpu_device_ids`. For each rank, the GDSManager receives only that rank's SSD file path as a one-device map, e.g. `c_ext.GDSManager({0: [rank_file_path]}, 1, round_robin=1)`. Do not create one shared GDSManager over all rank files and do not round-robin across rank files at GDSManager ownership granularity. Do not call `transfer_kv_blocks_gds`; its block-id abstraction is not compatible with this rank-file model. Instead, manually compute the file offset for each chunk using `chunk_offset = layer_id * layer_stride + is_v * kv_stride + blk_stride * blk_id`, then call `GDSManager.read(...)` for `DISK2D` or `GDSManager.write(...)` for `D2DISK` with an explicitly sliced GPU tensor and the computed byte offset. The worker API and tests must treat `block_ids` as rank-file block IDs: `rank_id` selects the rank file/GDSManager and `block_id` selects an offset inside that file. Do not expose or require local/global block terminology. Do not modify existing `GDSTransferWorker` or `tpGDSTransferWorker`; do not use explicit `tp_rank`; do not create C++ code.
  Parallelization: Wave 2 | Blocked by: Todos 1-2 | Blocks: benchmark integration and QA.
  References: `flexkv/transfer/worker.py:1778-2073`; `flexkv/transfer/worker.py:2076-2422`; `flexkv/transfer/worker.py:280-323`; `csrc/bindings.cpp:369-379`; `csrc/gds/gds_manager.cpp:303-340`; `csrc/gds/gds_manager.h:72-82`.
  Acceptance criteria (agent-executable): Static import check and a no-GDS-available smoke test that constructs the class with mocked `c_ext.GDSManager` and mocked `GDSManager.read/write`, then asserts it creates one GDSManager per `rank_id in range(len(tp_rank__to__file_path))` and each GDSManager is constructed with exactly one file as a one-device map, e.g. `{0: [rank_file_path]}`. The mock assertion must verify `DISK2D` calls `.read(rank_file_path, gpu_tensor_slice, chunk_offset_bytes)` and `D2DISK` calls `.write(rank_file_path, gpu_tensor_slice, chunk_offset_bytes)` for manually computed chunk offsets, with no `transfer_kv_blocks_gds` call. The smoke must also assert the worker derives `tp_group_size` from `tp_rank__to__file_path` and `num_blocks_per_file` from `ssd_layout.num_block`.
  QA scenarios: happy + failure, Evidence `.omo/evidence/task-3-rank-sharded-gds-phase1.md`. Happy: `DISK2D` and `D2DISK` both loop over ranks derived from metadata, create one rank-local GDSManager per rank, and call the correct read/write method with direct chunk offsets. Failure: any `transfer_kv_blocks_gds` call, missing required metadata, or local/global block mapping helper remains.
  Commit: Y | `feat(transfer): add rank-sharded GDS worker`

- [x] 4. Add rank-sharded GPU-SSD worker construction helper
  What to do / Must NOT do: Add `create_rank_sharded_gpu_ssd_worker(...)` in `benchmarks/benchmark_workers.py` that allocates per-rank GPU handles, rank-sharded SSD files via regular `SSDAllocator.allocate(...)` calls, symmetric rank layouts, op buffer, and returns `(RankShardedGDSTransferWorker, finished_ops_queue)`. Do not touch `TransferEngine` or production worker maps. Do not pass `tp_rank`; use `model_config.tp_size` and rank order.
  Parallelization: Wave 2 | Blocked by: Todos 1-3 | Blocks: benchmark CLI integration.
  References: `benchmarks/benchmark_workers.py:229-323`; `benchmarks/benchmark_workers.py:132-170`; `flexkv/common/storage.py:154-199`; `flexkv/common/config.py:611-650`; `flexkv/transfer/worker.py:280-323`.
  Acceptance criteria (agent-executable): Import-level test monkeypatches the benchmark-local allocation functions and calls the helper with a temporary cache dir, `tp_size=2`, `num_devices=2`, `num_ssd_blocks=4`, `num_gpu_blocks=2`, `num_layers=1`, and `dtype=torch.bfloat16`; assert returned worker has `tp_group_size=2`, one file per rank, `num_blocks_per_file=4`, and `rank_file_paths` matching Todo 2.
  QA scenarios: happy + failure, Evidence `.omo/evidence/task-4-rank-sharded-gds-phase1.md`. Happy: helper rejects empty cache dirs and non-divisible `num_head % tp_size != 0` for non-MLA. Failure: missing SSD cache dir raises a clear error.
  Commit: Y | `feat(bench): add rank-sharded GPU-SSD worker factory`

- [x] 5. Add benchmark CLI flag and dispatch path
  What to do / Must NOT do: Add a `--rank-sharded-gds` CLI flag to `benchmark_workers.py`. When enabled with `--transfer-type DISK2D` or `--transfer-type D2DISK`, use the new rank-sharded worker helper. Existing non-rank-sharded behavior must remain unchanged. Do not change `TransferType` enum or `TransferEngine`.
  Parallelization: Wave 3 | Blocked by: Todo 4 | Blocks: end-to-end benchmark QA.
  References: `benchmarks/benchmark_workers.py:43-70`; `benchmarks/benchmark_workers.py:435-477`; `benchmarks/benchmark_workers.py:580-667`; `flexkv/common/transfer.py:15-92`.
  Acceptance criteria (agent-executable): Run `python benchmarks/benchmark_workers.py --help` and assert `--rank-sharded-gds` appears. In a unit test, monkeypatch `transfer_kv_blocks_gds=None`, call `bench_worker` with a synthetic args object for `--rank-sharded-gds --transfer-type DISK2D`, and assert it returns `[]` and prints `[BENCH] GDS not compiled, skipping DISK2D/D2DISK`.
  QA scenarios: happy + failure, Evidence `.omo/evidence/task-5-rank-sharded-gds-phase1.md`. Happy: `--rank-sharded-gds` selects the new helper. Failure: `--rank-sharded-gds` with `H2D`/`D2H` raises a clear unsupported-transfer error.
  Commit: Y | `feat(bench): add rank-sharded GDS CLI path`

- [x] 6. Add focused unit tests for per-rank SSD placement, rank block id semantics, and worker call sequence
  What to do / Must NOT do: Add focused unit tests in `tests/test_rank_sharded_gds.py` that verify per-rank regular-SSD file placement, symmetric rank block id semantics, and worker call sequence without requiring real GPUs unless GDS is compiled. Tests must not use local/global block terminology except to assert that such public concepts are absent. Do not add unrelated production tests.
  Parallelization: Wave 3 | Blocked by: Todos 1-5 | Blocks: final verification.
  References: `tests/test_gds_get_planning.py:3-7`; `tests/test_kv_transfer_correctness.py:1-80`; `benchmarks/benchmark_workers.py:580-667`.
  Acceptance criteria (agent-executable): Run `pytest tests/test_rank_sharded_gds.py` and assert all mapping/allocation tests pass and the test runner reports success without requiring real GDS hardware.
  QA scenarios: happy + failure, Evidence `.omo/evidence/task-6-rank-sharded-gds-phase1.md`. Happy: pytest unit tests pass without GDS compiled. Failure: a changed mapping formula causes an assertion failure with the expected rank/file mapping.
  Commit: Y | `test: add rank-sharded GDS unit coverage`

- [x] 7. Run optional real GDS smoke benchmark when available
  What to do / Must NOT do: If `transfer_kv_blocks_gds` is compiled and GPUs/SSD are available, run a minimal `DISK2D` benchmark with `--rank-sharded-gds`, `--num-blocks 1`, one warmup, and one benchmark round. Do not require this in CI if GDS is unavailable.
  Parallelization: Wave 4 | Blocked by: Todos 1-6 | Blocks: final verification.
  References: `benchmarks/benchmark_workers.py:468-477`; `benchmarks/benchmark_workers.py:633-635`; `flexkv/transfer/worker.py:1812-1817`; `csrc/gds/gds_manager.cpp:618-620`.
  Acceptance criteria (agent-executable): If GDS is unavailable, record skip evidence. If available, run `python benchmarks/benchmark_workers.py --transfer-type DISK2D --rank-sharded-gds --config benchmarks/example_config.yml --num-blocks 1 --warmup-round 1 --benchmark-round 1` and assert the command exits 0 and prints `Avg Bandwidth`.
  QA scenarios: happy + failure, Evidence `.omo/evidence/task-7-rank-sharded-gds-phase1.md`. Happy: real GDS smoke completes. Failure: GDS unavailable or GPU/SSD setup invalid produces a recorded skip/failure reason, not a silent pass.
  Commit: Y | `test(bench): add optional rank-sharded GDS smoke path`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [x] F1. Plan compliance audit
- [x] F2. Code quality review
- [x] F3. Real manual QA
- [x] F4. Scope fidelity

## Commit strategy

- Keep product-code changes limited to `flexkv/transfer/worker.py`, `benchmarks/benchmark_workers.py`, `benchmarks/test_benchmark_workers.py`, and `tests/test_rank_sharded_gds.py`; `flexkv/storage/allocator.py` should remain unchanged.
- Prefer one implementation commit per todo, or squash into one final commit before execution: `feat(rank-sharded-gds): add worker and benchmark path`.
- Do not include `.omo/` plan artifacts in the product-code commit unless the workspace convention explicitly tracks them.
- Before commit, verify `git diff --name-only` contains only intended worker/benchmark/test changes plus any required generated evidence outside the commit boundary.

## Success criteria

- `benchmarks/benchmark_workers.py` uses regular `SSDAllocator` once per TP rank and does not add or require a dedicated rank-sharded allocator.
- `flexkv/transfer/worker.py` contains a new rank-sharded GDS worker that creates one existing `c_ext.GDSManager` per tp rank, passes only that rank's SSD file into each GDSManager, and uses direct `GDSManager.read/write` calls with manually computed chunk offsets.
- Existing `StorageEngine`, `TransferEngine`, `GDSTransferWorker`, and `tpGDSTransferWorker` behavior is unchanged.
- Rank IDs are always derived from `range(tp_size)`; no explicit `tp_rank` parameter exists in Phase1.
- The benchmark helper exposes minimal per-rank file metadata via `tp_rank__to__file_path`; a `block_id` is the offset inside the selected rank file, there is no rank-to-global-block mapping helper, and no dedicated allocator metadata is required.
- `benchmarks/benchmark_workers.py --help` exposes `--rank-sharded-gds`.
- `git diff --name-only` contains no changes to `StorageEngine`, `TransferEngine`, or the existing `SSDAllocator` implementation.
- `pytest tests/test_rank_sharded_gds.py` exits 0 and validates per-rank SSD placement, block mapping, and worker call sequence without real GDS hardware.
- Optional real GDS smoke is skipped with recorded evidence when GDS/GPU/SSD is unavailable; when available, the `DISK2D` smoke exits 0 and prints `Avg Bandwidth`.
