# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for benchmark worker CLI dispatch."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import pytest

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import benchmark_workers  # noqa: E402
from benchmark_workers import BenchmarkConfig, bench_worker, parse_args  # noqa: E402
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.transfer import TransferType

pytestmark = pytest.mark.unit


def _rank_sharded_config(transfer_type: TransferType) -> Tuple[ModelConfig, CacheConfig, BenchmarkConfig]:
    model_config = ModelConfig()
    cache_config = CacheConfig(enable_ssd=True, num_ssd_blocks=1, ssd_cache_dir="/tmp")
    bench_config = BenchmarkConfig(
        transfer_type=transfer_type,
        num_blocks_to_transfer=1,
        rank_sharded_gds=True,
    )
    return model_config, cache_config, bench_config


def test_parse_args_exposes_rank_sharded_gds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark_workers.py", "--rank-sharded-gds"])

    args = parse_args()

    assert args.rank_sharded_gds is True


def test_bench_worker_rank_sharded_gds_skips_when_gds_not_compiled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        benchmark_workers,
        "make_configs",
        lambda args: _rank_sharded_config(TransferType.DISK2D),
    )
    monkeypatch.setattr(benchmark_workers, "transfer_kv_blocks_gds", None)
    monkeypatch.setattr(benchmark_workers.torch.cuda, "device_count", lambda: 1)

    result = bench_worker(SimpleNamespace(rank_sharded_gds=True, sweep_blocks=None))

    assert result == []
    assert "[BENCH] GDS not compiled, skipping DISK2D/D2DISK" in capsys.readouterr().out


def test_bench_worker_rank_sharded_gds_rejects_unsupported_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_if_gpu_count_called() -> None:
        raise AssertionError("torch.cuda.device_count should not be called")

    monkeypatch.setattr(
        benchmark_workers,
        "make_configs",
        lambda args: _rank_sharded_config(TransferType.H2D),
    )
    monkeypatch.setattr(benchmark_workers.torch.cuda, "device_count", raise_if_gpu_count_called)

    with pytest.raises(
        ValueError,
        match="rank-sharded GDS benchmark only supports DISK2D/D2DISK transfer types; got H2D",
    ):
        bench_worker(SimpleNamespace(rank_sharded_gds=True, sweep_blocks=None))
