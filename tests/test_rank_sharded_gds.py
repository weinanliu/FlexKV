import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
for path in (REPO_ROOT, BENCHMARKS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import benchmarks.benchmark_workers as benchmark_workers_module
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferOp, TransferType
import flexkv.transfer.worker as worker_module


pytestmark = pytest.mark.unit


class _FakeTensorHandle:
    def __init__(self, tensor):
        self._tensor = tensor

    def get_tensor(self):
        return self._tensor


class _FakeStorageHandle:
    def __init__(self, tensor_handles, kv_layout):
        self._tensor_handles = tensor_handles
        self.kv_layout = kv_layout

    def get_tensor_handle_list(self):
        return self._tensor_handles


class _FakeSSDStorageHandle:
    def __init__(self, file_path, kv_layout):
        self._file_path = file_path
        self.kv_layout = kv_layout

    def get_file_list(self):
        return {0: [str(self._file_path)]}


class _FakeCudaStream:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _NoopStreamContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeGDSManager:
    instances = []
    calls = []

    def __init__(self, file_map, num_devices, round_robin=None):
        self.file_map = file_map
        self.num_devices = num_devices
        self.round_robin = round_robin
        self.calls = []
        self.instance_id = len(_FakeGDSManager.instances)
        _FakeGDSManager.instances.append(self)

    def is_ready(self):
        return True

    def read(self, file_path, gpu_tensor_slice, chunk_offset_bytes):
        call = ("read", file_path, gpu_tensor_slice, chunk_offset_bytes)
        self.calls.append(call)
        _FakeGDSManager.calls.append(call)

    def write(self, file_path, gpu_tensor_slice, chunk_offset_bytes):
        call = ("write", file_path, gpu_tensor_slice, chunk_offset_bytes)
        self.calls.append(call)
        _FakeGDSManager.calls.append(call)


def _reset_fake_gds():
    _FakeGDSManager.instances.clear()
    _FakeGDSManager.calls.clear()


def _make_configs(cache_dir, *, tp_size=2, num_gpu_blocks=2, num_ssd_blocks=4, num_layers=1):
    model_config = ModelConfig(
        num_layers=num_layers,
        num_kv_heads=2,
        head_size=8,
        use_mla=False,
        dtype=torch.bfloat16,
        tp_size=tp_size,
    )
    cache_config = CacheConfig(
        tokens_per_block=4,
        num_ssd_blocks=num_ssd_blocks,
        ssd_cache_dir=str(cache_dir),
    )
    return model_config, cache_config


def _make_layouts(*, num_gpu_blocks, num_ssd_blocks, num_layers):
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=4,
        num_head=1,
        head_size=8,
        is_mla=False,
    )
    ssd_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers,
        num_block=num_ssd_blocks,
        tokens_per_block=4,
        num_head=1,
        head_size=8,
        is_mla=False,
    )
    return gpu_layout, ssd_layout


def _install_cuda_monkeypatches(monkeypatch, set_device_calls):
    def fake_set_device(device_id):
        set_device_calls.append(int(device_id))

    monkeypatch.setattr(worker_module, "ensure_cuda_device", lambda device: None)
    monkeypatch.setattr(
        worker_module,
        "import_tensor_handles",
        lambda handles: [handle.get_tensor() for handle in handles],
    )
    monkeypatch.setattr(worker_module, "c_ext", SimpleNamespace(GDSManager=_FakeGDSManager))
    monkeypatch.setattr(torch.cuda, "set_device", fake_set_device)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: _FakeCudaStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _NoopStreamContext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _install_allocator_monkeypatches(monkeypatch, tmp_path, gpu_allocations, ssd_allocations):
    def fake_gpu_allocate(layout, dtype, **kwargs):
        device_id = int(kwargs["device_id"])
        gpu_allocations.append((layout, dtype, dict(kwargs)))
        tensor = torch.arange(layout.get_total_elements(), dtype=dtype)
        return _FakeStorageHandle([_FakeTensorHandle(tensor)], layout)

    def fake_ssd_allocate(layout, dtype, **kwargs):
        cache_dir = Path(kwargs["cache_dir"])
        file_prefix = kwargs["file_prefix"]
        rank_id = len(ssd_allocations)
        file_path = cache_dir / f"{file_prefix}_{0}_{0}.bin"
        file_path.write_bytes(b"")
        ssd_allocations.append((layout, dtype, dict(kwargs), file_path))
        return _FakeSSDStorageHandle(file_path, layout)

    monkeypatch.setattr(benchmark_workers_module.GPUAllocator, "allocate", fake_gpu_allocate)
    monkeypatch.setattr(benchmark_workers_module.SSDAllocator, "allocate", fake_ssd_allocate)


def _expected_rank_file_paths(tmp_path):
    return [
        str(tmp_path / "flexkv_rank_sharded_ssd_cache_rank_0_0_0.bin"),
        str(tmp_path / "flexkv_rank_sharded_ssd_cache_rank_1_0_0.bin"),
    ]


def _expected_chunk_offsets_bytes(layout, dtype, block_id):
    return [
        (
            0 * layout.get_layer_stride()
            + is_v * layout.get_kv_stride()
            + block_id * layout.get_block_stride()
        )
        * dtype.itemsize
        for is_v in range(layout.kv_dim)
    ]


def _expected_gpu_tensor_offsets(layout, block_id):
    return [
        0 * layout.get_layer_stride()
        + is_v * layout.get_kv_stride()
        + block_id * layout.get_block_stride()
        for is_v in range(layout.kv_dim)
    ]


def _close_queue(queue):
    queue.close()
    queue.join_thread()


def test_create_rank_sharded_gpu_ssd_worker_allocates_per_rank_files_and_metadata(
    tmp_path,
    monkeypatch,
):
    _reset_fake_gds()
    model_config, cache_config = _make_configs(tmp_path)
    gpu_allocations = []
    ssd_allocations = []
    set_device_calls = []
    _install_cuda_monkeypatches(monkeypatch, set_device_calls)
    _install_allocator_monkeypatches(monkeypatch, tmp_path, gpu_allocations, ssd_allocations)

    worker, finished_ops_queue = benchmark_workers_module.create_rank_sharded_gpu_ssd_worker(
        model_config=model_config,
        cache_config=cache_config,
        num_gpu_blocks=2,
        gpu_layout_type=0,
    )

    try:
        expected_paths = _expected_rank_file_paths(tmp_path)
        assert len(gpu_allocations) == 2
        assert [allocation[2]["device_id"] for allocation in gpu_allocations] == [0, 1]
        assert [allocation[1] for allocation in gpu_allocations] == [torch.bfloat16, torch.bfloat16]

        assert len(ssd_allocations) == 2
        assert [str(allocation[3]) for allocation in ssd_allocations] == expected_paths
        assert [allocation[2]["num_chunks"] for allocation in ssd_allocations] == [1, 1]
        assert all(Path(file_path).exists() for file_path in expected_paths)

        assert worker.tp_group_size == 2
        assert worker.num_blocks_per_file == 4
        assert worker.rank_file_paths == expected_paths
        assert worker.ssd_layout.num_block == 4
        assert worker.gpu_kv_layouts[0].num_block == 2
        assert worker.gpu_kv_layouts[0].num_head == 1
        assert len(worker.gds_managers) == 2
        assert set_device_calls == [0, 1, 0, 1]
    finally:
        _close_queue(finished_ops_queue)


def test_rank_sharded_gds_worker_uses_per_rank_managers_and_manual_offsets(
    tmp_path,
    monkeypatch,
):
    _reset_fake_gds()
    model_config, cache_config = _make_configs(tmp_path)
    gpu_allocations = []
    ssd_allocations = []
    set_device_calls = []
    _install_cuda_monkeypatches(monkeypatch, set_device_calls)
    _install_allocator_monkeypatches(monkeypatch, tmp_path, gpu_allocations, ssd_allocations)

    worker, finished_ops_queue = benchmark_workers_module.create_rank_sharded_gpu_ssd_worker(
        model_config=model_config,
        cache_config=cache_config,
        num_gpu_blocks=2,
        gpu_layout_type=0,
    )

    try:
        expected_paths = _expected_rank_file_paths(tmp_path)
        assert len(_FakeGDSManager.instances) == 2
        assert [instance.file_map for instance in _FakeGDSManager.instances] == [
            {0: [expected_paths[0]]},
            {0: [expected_paths[1]]},
        ]
        assert [instance.num_devices for instance in _FakeGDSManager.instances] == [1, 1]
        assert [instance.round_robin for instance in _FakeGDSManager.instances] == [1, 1]
        assert worker.gds_managers[0] is not worker.gds_managers[1]

        disk2d_op = TransferOp(
            graph_id=11,
            transfer_type=TransferType.DISK2D,
            src_block_ids=np.array([1], dtype=np.int64),
            dst_block_ids=np.array([0], dtype=np.int64),
        )
        assert not np.array_equal(disk2d_op.src_block_ids, disk2d_op.dst_block_ids)
        worker.submit_transfer(disk2d_op)
        assert finished_ops_queue.get(timeout=1.0) == (disk2d_op.op_id, True, None)

        d2disk_op = TransferOp(
            graph_id=12,
            transfer_type=TransferType.D2DISK,
            src_block_ids=np.array([0], dtype=np.int64),
            dst_block_ids=np.array([1], dtype=np.int64),
        )
        assert not np.array_equal(d2disk_op.src_block_ids, d2disk_op.dst_block_ids)
        worker.submit_transfer(d2disk_op)
        assert finished_ops_queue.get(timeout=1.0) == (d2disk_op.op_id, True, None)

        expected_offsets_bytes = _expected_chunk_offsets_bytes(
            worker.ssd_layout,
            torch.bfloat16,
            block_id=1,
        )
        expected_gpu_tensor_offsets = _expected_gpu_tensor_offsets(worker.gpu_kv_layouts[0], block_id=0)

        for rank_id, rank_file_path in enumerate(expected_paths):
            rank_calls = [
                call for call in _FakeGDSManager.calls
                if call[1] == rank_file_path
            ]
            assert [(call[0], call[3]) for call in rank_calls] == [
                ("read", expected_offsets_bytes[0]),
                ("read", expected_offsets_bytes[1]),
                ("write", expected_offsets_bytes[0]),
                ("write", expected_offsets_bytes[1]),
            ]
            assert [call[2].shape for call in rank_calls] == [
                (worker.ssd_layout.get_chunk_size(),),
                (worker.ssd_layout.get_chunk_size(),),
                (worker.ssd_layout.get_chunk_size(),),
                (worker.ssd_layout.get_chunk_size(),),
            ]
            assert [call[2].storage_offset() for call in rank_calls] == (
                expected_gpu_tensor_offsets + expected_gpu_tensor_offsets
            )

        bad_op = TransferOp(
            graph_id=13,
            transfer_type=TransferType.DISK2D,
            src_block_ids=np.array([4], dtype=np.int64),
            dst_block_ids=np.array([0], dtype=np.int64),
        )
        call_count_before_bad_op = len(_FakeGDSManager.calls)
        worker.submit_transfer(bad_op)
        assert finished_ops_queue.get(timeout=1.0) == (bad_op.op_id, False, None)
        assert len(_FakeGDSManager.calls) == call_count_before_bad_op

        bad_gpu_op = TransferOp(
            graph_id=14,
            transfer_type=TransferType.DISK2D,
            src_block_ids=np.array([1], dtype=np.int64),
            dst_block_ids=np.array([4], dtype=np.int64),
        )
        call_count_before_bad_gpu_op = len(_FakeGDSManager.calls)
        worker.submit_transfer(bad_gpu_op)
        assert finished_ops_queue.get(timeout=1.0) == (bad_gpu_op.op_id, False, None)
        assert len(_FakeGDSManager.calls) == call_count_before_bad_gpu_op
    finally:
        _close_queue(finished_ops_queue)


def test_rank_sharded_gds_worker_source_does_not_expose_rank_file_block_mapping_terms():
    worker_source = (Path(__file__).resolve().parents[1] / "flexkv/transfer/worker.py").read_text()
    assert "local_block" not in worker_source
    assert "global_block" not in worker_source
