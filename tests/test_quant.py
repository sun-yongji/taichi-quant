"""
Tests for TaiChi-Quant: C6 Symmetry-Driven Quantization Engine.
"""

import numpy as np
import pytest

from taichi_quant import (
    QuantEngine,
    LayerProfiler,
    QuantLevel,
    QuantStrategy,
    QuantConfig,
    LayerProfile,
    simulate_quantize,
    quick_assess,
    QUANT_BIT_MAP,
    QUANT_NAME_MAP,
    HEX_COUPLING,
    MAX_GROUPS,
)


class TestConstants:
    def test_quant_bit_map(self):
        assert QUANT_BIT_MAP == {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8}

    def test_quant_name_map(self):
        assert QUANT_NAME_MAP[0] == "Q2_K"
        assert QUANT_NAME_MAP[5] == "Q8_0"

    def test_max_groups(self):
        assert MAX_GROUPS == 6

    def test_coupling_shape(self):
        assert HEX_COUPLING.shape == (6, 6)


class TestQuantLevel:
    def test_bits(self):
        assert QuantLevel.Q2_K.bits == 2
        assert QuantLevel.Q3_K.bits == 3
        assert QuantLevel.Q4_K.bits == 4
        assert QuantLevel.Q5_K.bits == 5
        assert QuantLevel.Q6_K.bits == 6
        assert QuantLevel.Q8_0.bits == 8

    def test_name(self):
        assert QuantLevel.Q4_K.name == "Q4_K"
        assert QuantLevel.Q8_0.name == "Q8_0"

    def test_from_bits(self):
        assert QuantLevel.from_bits(4) == QuantLevel.Q4_K
        assert QuantLevel.from_bits(8) == QuantLevel.Q8_0
        with pytest.raises(ValueError):
            QuantLevel.from_bits(7)

    def test_repr(self):
        assert "Q8_0" in repr(QuantLevel.Q8_0)


class TestLayerProfiler:
    @pytest.fixture
    def profiler(self):
        return LayerProfiler(n_bins=128)

    def test_profile_basic(self, profiler):
        w = np.random.randn(100, 200)
        p = profiler.profile("test_layer", w)
        assert p.name == "test_layer"
        assert p.shape == (100, 200)
        assert 0 <= p.entropy <= 1
        assert 0 <= p.sparsity <= 1
        assert 0 <= p.outlier_ratio <= 1

    def test_profile_constant(self, profiler):
        # Constant tensor → max sparsity, no outliers, uniform entropy
        w = np.ones((50, 50))
        p = profiler.profile("const", w)
        assert p.outlier_ratio < 0.01

    def test_profile_outliers(self, profiler):
        # One extreme outlier
        w = np.random.randn(50, 50) * 0.1
        w[0, 0] = 100.0
        p = profiler.profile("outlier", w)
        assert p.outlier_ratio > 0.0

    def test_profile_many(self, profiler):
        tensors = {
            "a": np.random.randn(32, 64),
            "b": np.random.randn(64, 128),
            "c": np.random.randn(128, 256),
        }
        profiles = profiler.profile_many(tensors)
        assert len(profiles) == 3
        assert all(isinstance(p, LayerProfile) for p in profiles)

    def test_coupling_index_cycling(self, profiler):
        tensors = {f"layer.{i}": np.random.randn(16, 16) for i in range(12)}
        profiles = profiler.profile_many(tensors)
        indices = [p.coupling_index for p in profiles]
        assert indices == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]


class TestQuantEngine:
    @pytest.fixture
    def engine(self):
        return QuantEngine()

    @pytest.fixture
    def sample_profiles(self, engine):
        tensors = {
            "embed": np.random.randn(1000, 512) * 0.01,
            "attn.q.0": np.random.randn(512, 512) * 0.12,
            "mlp.0": np.random.randn(2048, 512) * 0.15,
            "attn.q.1": np.random.randn(512, 512) * 0.10,
            "mlp.1": np.random.randn(2048, 512) * 0.17,
            "output": np.random.randn(512, 1000) * 0.05,
        }
        return engine.profiler.profile_many(tensors)

    def test_suggest_level(self, engine):
        # Low sensitivity → aggressive quant
        p = LayerProfile("test", (32, 32), 0.0, 0.01, -0.1, 0.1, 0.3, 0.9, 0.0, 0)
        level = engine.suggest_level(p)
        assert level.bits <= 4

    def test_suggest_high_sensitivity(self, engine):
        # High outlier → preserve
        p = LayerProfile("test", (32, 32), 0.0, 1.0, -10, 10, 0.5, 0.0, 0.5, 0)
        level = engine.suggest_level(p)
        assert level.bits >= 6

    def test_strategize_returns_config(self, engine, sample_profiles):
        config = engine.strategize(sample_profiles)
        assert isinstance(config, QuantConfig)
        assert len(config.strategies) >= 1
        assert config.total_params > 0
        assert config.compressed_size_bits < config.original_size_bits
        assert config.overall_compression > 1.0

    def test_strategize_compression(self, engine, sample_profiles):
        config = engine.strategize(sample_profiles, original_bits=16)
        # With mixed quantization, compression should be between 2x and 8x
        assert 1.5 < config.overall_compression < 10

    def test_strategize_groups(self, engine, sample_profiles):
        config = engine.strategize(sample_profiles)
        # 7 layers with i%6 cycling → all 6 groups used
        assert len(config.group_summary) >= 5

    def test_empty_profiles(self, engine):
        config = engine.strategize([])
        assert len(config.strategies) == 0

    def test_repr(self, engine):
        r = repr(engine)
        assert "QuantEngine" in r


class TestSimulateQuantize:
    def test_q2k_mean_error(self):
        w = np.random.randn(100, 100)
        _, mae = simulate_quantize(w, QuantLevel.Q2_K)
        # With 2 bits, MAE should be noticeable
        assert mae > 0

    def test_q8_0_low_error(self):
        w = np.random.randn(100, 100)
        _, mae = simulate_quantize(w, QuantLevel.Q8_0)
        # 8 bits → very low error
        assert mae < 0.1

    def test_higher_bits_lower_error(self):
        w = np.random.randn(100, 100)
        _, mae_q2 = simulate_quantize(w, QuantLevel.Q2_K)
        _, mae_q8 = simulate_quantize(w, QuantLevel.Q8_0)
        assert mae_q8 < mae_q2

    def test_constant_tensor(self):
        w = np.ones((50, 50))
        w_q, mae = simulate_quantize(w, QuantLevel.Q4_K)
        assert mae == 0.0
        assert np.allclose(w, w_q)

    def test_shape_preserved(self):
        w = np.random.randn(32, 64, 3)
        w_q, _ = simulate_quantize(w, QuantLevel.Q6_K)
        assert w_q.shape == w.shape


class TestQuickAssess:
    def test_basic(self):
        tensors = {
            "a": np.random.randn(64, 128),
            "b": np.random.randn(128, 256),
        }
        results = quick_assess(tensors)
        assert len(results) == 2
        assert all("level" in r for r in results)
        assert all("mae" in r for r in results)
        assert all("bits" in r for r in results)

    def test_empty(self):
        results = quick_assess({})
        assert results == []


class TestQuantConfig:
    def test_fields(self):
        config = QuantConfig(
            strategies=[],
            total_params=1000,
            compressed_size_bits=2000,
            original_size_bits=16000,
            overall_compression=8.0,
            avg_fidelity_loss=0.1,
        )
        assert config.overall_compression == 8.0
