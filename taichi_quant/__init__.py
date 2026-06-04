"""
TaiChi-Quant: C6 Symmetry-Driven Quantization Tool

Maps the six hexagram yao positions to six GGUF quantization levels
(Q2_K→Q8_0), using hexagonal coupling for layer-group sharing.

Key components:
- QuantEngine: strategy generation from layer sensitivity profiles
- LayerProfiler: weight tensor sensitivity analysis
- simulate_quantize: quantization error estimation
- quick_assess: rapid model-wide quantization assessment
"""

from .quantizer import (
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
    PHI_COMP,
    MAX_GROUPS,
)

__all__ = [
    "QuantEngine",
    "LayerProfiler",
    "QuantLevel",
    "QuantStrategy",
    "QuantConfig",
    "LayerProfile",
    "simulate_quantize",
    "quick_assess",
    "QUANT_BIT_MAP",
    "QUANT_NAME_MAP",
    "HEX_COUPLING",
    "PHI_COMP",
    "MAX_GROUPS",
]

__version__ = "0.1.0"
