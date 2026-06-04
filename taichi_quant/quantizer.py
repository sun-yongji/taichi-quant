"""
TaiChi-Quant: C6 Symmetry-Driven Quantization Tool

Maps the six hexagram yao positions to six GGUF quantization levels,
using hexagonal coupling to determine layer-group sharing strategies.

Quantization levels (初→上):
  Q2_K → 初爻 (deepest compression, 2-bit)
  Q3_K → 二爻 (3-bit)
  Q4_K → 三爻 (4-bit)
  Q5_K → 四爻 (5-bit)
  Q6_K → 五爻 (6-bit)
  Q8_0 → 上爻 (lightest compression, 8-bit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Six quantization levels ↔ six yao positions (初→上)
QUANT_BIT_MAP: Dict[int, int] = {
    0: 2,  # Q2_K → 初爻
    1: 3,  # Q3_K → 二爻
    2: 4,  # Q4_K → 三爻
    3: 5,  # Q5_K → 四爻
    4: 6,  # Q6_K → 五爻
    5: 8,  # Q8_0 → 上爻
}

QUANT_NAME_MAP: Dict[int, str] = {
    0: "Q2_K",
    1: "Q3_K",
    2: "Q4_K",
    3: "Q5_K",
    4: "Q6_K",
    5: "Q8_0",
}

# C6 hexagonal coupling matrix (same as router)
HEX_COUPLING: np.ndarray = np.array([
    [1.000, 0.500, 0.000, 0.000, 0.000, 0.500],
    [0.500, 1.000, 0.500, 0.000, 0.000, 0.000],
    [0.000, 0.500, 1.000, 0.500, 0.000, 0.000],
    [0.000, 0.000, 0.500, 1.000, 0.500, 0.000],
    [0.000, 0.000, 0.000, 0.500, 1.000, 0.500],
    [0.500, 0.000, 0.000, 0.000, 0.500, 1.000],
], dtype=np.float64)

# Gold-ratio compression factor
PHI_COMP: float = 0.618

# Maximum quantization groups (hexagram-based)
MAX_GROUPS: int = 6


class QuantLevel(Enum):
    """Quantization level from Q2_K (deepest) to Q8_0 (lightest)."""
    Q2_K = 0
    Q3_K = 1
    Q4_K = 2
    Q5_K = 3
    Q6_K = 4
    Q8_0 = 5

    @property
    def bits(self) -> int:
        return QUANT_BIT_MAP[self.value]

    @property
    def name(self) -> str:
        return QUANT_NAME_MAP[self.value]

    @classmethod
    def from_bits(cls, bits: int) -> "QuantLevel":
        for level in cls:
            if level.bits == bits:
                return level
        raise ValueError(f"No quantization level for {bits} bits")

    def __repr__(self) -> str:
        return f"QuantLevel.{self.name}"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerProfile:
    """Sensitivity profile for a single layer."""
    name: str
    shape: Tuple[int, ...]
    # Weight statistics
    weight_mean: float
    weight_std: float
    weight_min: float
    weight_max: float
    # Sensitivity metrics
    entropy: float          # Normalized entropy of weight distribution
    sparsity: float         # Fraction of near-zero weights
    outlier_ratio: float    # Fraction of values > 3σ from mean
    # Coupling index into hexagonal group (0-5)
    coupling_index: int = 0


@dataclass
class QuantStrategy:
    """Quantization strategy for one layer or layer group."""
    # Which layers share this strategy
    layer_names: List[str]
    # Assigned quantization level
    level: QuantLevel
    # Whether this group shares quantization parameters
    shared_params: bool = False
    # Hexagonal coupling group (0-5)
    hex_group: int = -1
    # Estimated compression ratio
    compression_ratio: float = 0.0
    # Estimated fidelity loss (0-1, lower is better)
    fidelity_loss: float = 0.0


@dataclass
class QuantConfig:
    """Complete quantization configuration for a model."""
    strategies: List[QuantStrategy]
    # Overall stats
    total_params: int
    compressed_size_bits: int
    original_size_bits: int
    overall_compression: float
    avg_fidelity_loss: float
    # Hexagonal grouping summary
    group_summary: Dict[int, List[str]] = field(default_factory=dict)
    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer profiler
# ---------------------------------------------------------------------------

class LayerProfiler:
    """Profile layer tensors for sensitivity analysis.

    Computes entropy, sparsity, and outlier metrics to determine
    how aggressively each layer can be quantized.
    """

    def __init__(self, n_bins: int = 256):
        self.n_bins = n_bins

    def profile(
        self,
        name: str,
        tensor: np.ndarray,
        coupling_index: int = 0,
    ) -> LayerProfile:
        """Profile a single layer's weight tensor.

        Args:
            name: Layer name (e.g., "model.layers.0.mlp.down_proj")
            tensor: Weight tensor (any shape, flattened for analysis)
            coupling_index: Hexagonal coupling group index (0-5)

        Returns:
            LayerProfile with sensitivity metrics
        """
        w = np.asarray(tensor, dtype=np.float64).ravel()
        n = len(w)

        # Basic statistics
        w_mean = float(np.mean(w))
        w_std = float(np.std(w))
        w_min = float(np.min(w))
        w_max = float(np.max(w))

        # Entropy: discretize into bins, compute normalized entropy
        hist, _ = np.histogram(w, bins=self.n_bins)
        hist = hist.astype(np.float64) / n  # normalize to probabilities
        eps = 1e-12
        ent = -float(np.sum(hist * np.log(hist + eps)))
        ent_max = math.log(self.n_bins)
        entropy_norm = ent / ent_max if ent_max > 0 else 0.0

        # Sparsity: fraction of weights within ε of zero
        eps_zero = float(np.std(w)) * 0.01 if w_std > 0 else 1e-6
        sparsity = float(np.sum(np.abs(w) < eps_zero)) / n

        # Outlier ratio: fraction beyond 3σ
        if w_std > 0:
            outlier_ratio = float(np.sum(np.abs(w - w_mean) > 3.0 * w_std)) / n
        else:
            outlier_ratio = 0.0

        return LayerProfile(
            name=name,
            shape=tensor.shape,
            weight_mean=w_mean,
            weight_std=w_std,
            weight_min=w_min,
            weight_max=w_max,
            entropy=entropy_norm,
            sparsity=sparsity,
            outlier_ratio=outlier_ratio,
            coupling_index=coupling_index,
        )

    def profile_many(
        self,
        tensors: Dict[str, np.ndarray],
    ) -> List[LayerProfile]:
        """Profile multiple layers at once.

        Args:
            tensors: Dict of name → tensor

        Returns:
            List of LayerProfile objects
        """
        profiles = []
        for i, (name, tensor) in enumerate(tensors.items()):
            ci = i % MAX_GROUPS
            profiles.append(self.profile(name, tensor, coupling_index=ci))
        return profiles


# ---------------------------------------------------------------------------
# Quantization engine
# ---------------------------------------------------------------------------

class QuantEngine:
    """C6 symmetry-driven quantization engine.

    Assigns quantization levels to layers based on sensitivity profiling
    and hexagonal coupling group sharing.

    Usage::

        engine = QuantEngine()
        profiles = engine.profiler.profile_many(layer_dict)
        config = engine.strategize(profiles)
        for s in config.strategies:
            print(f"{s.layer_names} → {s.level.name} (loss={s.fidelity_loss:.3f})")
    """

    # Sensitivity thresholds for bit allocation
    # Low entropy + low outlier → safe for aggressive quantization
    SENS_HIGH = 0.7   # High sensitivity → preserve with Q8_0
    SENS_MED = 0.4    # Medium sensitivity → Q5_K or Q4_K
    # Below SENS_MED: low sensitivity → Q3_K or Q2_K

    def __init__(
        self,
        coupling_matrix: Optional[np.ndarray] = None,
        target_compression: float = 4.0,
        phi_comp: float = PHI_COMP,
    ):
        self.coupling_matrix = (
            coupling_matrix if coupling_matrix is not None
            else HEX_COUPLING.copy()
        )
        self.target_compression = target_compression
        self.phi_comp = phi_comp
        self.profiler = LayerProfiler()

    def _compute_sensitivity(self, profile: LayerProfile) -> float:
        """Compute overall sensitivity score (0-1, higher = more sensitive).

        Sensitive layers need higher-bit quantization to preserve fidelity.
        """
        # Outliers → high sensitivity (critical weights)
        # High entropy → moderate sensitivity (diverse distribution)
        # High sparsity → low sensitivity (many near-zero weights)
        s = (
            0.50 * profile.outlier_ratio +
            0.30 * (1.0 - profile.entropy) +  # low entropy = diverse = sensitive
            0.20 * (1.0 - profile.sparsity)     # dense = more to preserve
        )
        return float(np.clip(s, 0.0, 1.0))

    def _sensitivity_to_level(self, sensitivity: float) -> QuantLevel:
        """Map sensitivity score to quantization level."""
        if sensitivity > self.SENS_HIGH:
            return QuantLevel.Q8_0   # 8-bit, preserve sensitive
        elif sensitivity > self.SENS_MED:
            # Gold-ratio split between Q5_K and Q4_K
            if sensitivity > (self.SENS_MED + self.SENS_HIGH) / 2:
                return QuantLevel.Q6_K
            else:
                return QuantLevel.Q5_K
        else:
            # Low sensitivity: aggressive compression
            if sensitivity > self.SENS_MED * self.phi_comp:
                return QuantLevel.Q4_K
            elif sensitivity > self.SENS_MED * self.phi_comp * 0.5:
                return QuantLevel.Q3_K
            else:
                return QuantLevel.Q2_K

    def _group_layers(
        self,
        profiles: List[LayerProfile],
    ) -> Dict[int, List[LayerProfile]]:
        """Group layers by hexagonal coupling index.

        Layers in the same hex group can share quantization parameters,
        reducing metadata overhead.
        """
        groups: Dict[int, List[LayerProfile]] = {i: [] for i in range(MAX_GROUPS)}
        for p in profiles:
            groups[p.coupling_index].append(p)
        return groups

    def _fidelity_loss(self, level: QuantLevel, sensitivity: float) -> float:
        """Estimate fidelity loss for a given quantization level.

        Lower bits → higher loss, sensitivity amplifies loss.
        Loss = (1 - bits/8) * sensitivity * φ
        """
        bits = level.bits
        base_loss = 1.0 - (bits / 16.0)  # normalize against fp16
        return float(base_loss * sensitivity * self.phi_comp)

    def _compression_ratio(self, level: QuantLevel, original_bits: int = 16) -> float:
        """Compression ratio for a quantization level."""
        return original_bits / level.bits

    def strategize(
        self,
        profiles: List[LayerProfile],
        original_bits: int = 16,
    ) -> QuantConfig:
        """Generate a complete quantization strategy from layer profiles.

        Args:
            profiles: List of layer sensitivity profiles
            original_bits: Bits per weight in original model (default: fp16=16)

        Returns:
            QuantConfig with per-layer/per-group strategies
        """
        strategies: List[QuantStrategy] = []
        groups = self._group_layers(profiles)

        total_params = 0
        compressed_size = 0
        original_size = 0
        total_loss = 0.0
        n_layers = len(profiles)

        group_summary: Dict[int, List[str]] = {}

        for gid, gprofs in groups.items():
            if not gprofs:
                continue

            # Average sensitivity in group
            sensitivities = [self._compute_sensitivity(p) for p in gprofs]
            avg_sens = float(np.mean(sensitivities))

            # Group-level quantization level
            level = self._sensitivity_to_level(avg_sens)

            # Per-layer strategies (may share params within group)
            layer_names = [p.name for p in gprofs]
            group_summary[gid] = layer_names

            # Layers in same hex group with similar sensitivity share params
            shared = len(gprofs) > 1
            fl = self._fidelity_loss(level, avg_sens)
            cr = self._compression_ratio(level, original_bits)

            strategy = QuantStrategy(
                layer_names=layer_names,
                level=level,
                shared_params=shared,
                hex_group=gid,
                compression_ratio=round(cr, 2),
                fidelity_loss=round(fl, 4),
            )
            strategies.append(strategy)

            # Accumulate stats
            for p in gprofs:
                n = int(np.prod(p.shape))
                total_params += n
                compressed_size += n * level.bits
                original_size += n * original_bits
                total_loss += fl

        overall_cr = original_size / compressed_size if compressed_size > 0 else 1.0
        avg_loss = total_loss / n_layers if n_layers > 0 else 0.0

        return QuantConfig(
            strategies=strategies,
            total_params=total_params,
            compressed_size_bits=compressed_size,
            original_size_bits=original_size,
            overall_compression=round(overall_cr, 2),
            avg_fidelity_loss=round(avg_loss, 4),
            group_summary=group_summary,
            meta={
                "target_compression": self.target_compression,
                "n_layers": n_layers,
                "n_groups": len(strategies),
                "phi_comp": self.phi_comp,
            },
        )

    def suggest_level(
        self,
        profile: LayerProfile,
    ) -> QuantLevel:
        """Quick single-layer quantization suggestion."""
        return self._sensitivity_to_level(self._compute_sensitivity(profile))

    def __repr__(self) -> str:
        return (
            f"QuantEngine(target_compression={self.target_compression}, "
            f"phi_comp={self.phi_comp})"
        )


# ---------------------------------------------------------------------------
# Utility: simulate quantization
# ---------------------------------------------------------------------------

def simulate_quantize(
    tensor: np.ndarray,
    level: QuantLevel,
) -> Tuple[np.ndarray, float]:
    """Simulate quantization of a tensor at a given level.

    Returns (quantized_tensor, mean_absolute_error).

    This is a simplified simulation — packs values into 2^bits levels
    and decompresses, measuring the reconstruction error.
    """
    w = np.asarray(tensor, dtype=np.float64)
    bits = level.bits
    n_levels = 2 ** bits

    # Min-max normalization
    w_min, w_max = w.min(), w.max()
    if w_max - w_min < 1e-12:
        return w.copy(), 0.0

    # Quantize: scale to [0, n_levels-1], round, scale back
    scale = (n_levels - 1) / (w_max - w_min)
    w_q = np.round((w - w_min) * scale).astype(np.int32)
    w_q = np.clip(w_q, 0, n_levels - 1)

    # Dequantize
    w_dq = w_q.astype(np.float64) / scale + w_min

    mae = float(np.mean(np.abs(w - w_dq)))
    return w_dq, mae


def quick_assess(
    tensors: Dict[str, np.ndarray],
    engine: Optional[QuantEngine] = None,
) -> List[Dict[str, Any]]:
    """Quick assessment: profile + quantize all layers, report fidelity.

    Returns a list of dicts with per-layer assessment.
    """
    if engine is None:
        engine = QuantEngine()

    results = []
    for name, t in tensors.items():
        profile = engine.profiler.profile(name, t)
        level = engine.suggest_level(profile)
        _, mae = simulate_quantize(t, level)
        results.append({
            "name": name,
            "shape": profile.shape,
            "sensitivity": round(engine._compute_sensitivity(profile), 4),
            "level": level.name,
            "bits": level.bits,
            "mae": round(mae, 6),
            "entropy": round(profile.entropy, 4),
        })
    return results
