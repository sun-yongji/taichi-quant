"""
TaiChi-Quant demo: quantize simulated model layers and show strategy.
"""

import numpy as np

from taichi_quant import QuantEngine, quick_assess, QuantLevel, simulate_quantize


def main():
    print("=" * 60)
    print("  TaiChi-Quant: C6 Symmetry-Driven Quantization Demo")
    print("=" * 60)

    # Simulate a small transformer model's weight tensors
    rng = np.random.default_rng(42)
    tensors = {
        "tok_embeddings":       rng.normal(0, 0.02, (8000, 512)),
        "layer.0.attention.wq":  rng.normal(0, 0.08, (512, 512)),
        "layer.0.attention.wk":  rng.normal(0, 0.08, (512, 512)),
        "layer.0.attention.wv":  rng.normal(0, 0.08, (512, 512)),
        "layer.0.attention.wo":  rng.normal(0, 0.10, (512, 512)),
        "layer.0.mlp.w1":        rng.normal(0, 0.12, (2048, 512)),
        "layer.0.mlp.w2":        rng.normal(0, 0.10, (512, 2048)),
        "layer.0.mlp.w3":        rng.normal(0, 0.12, (2048, 512)),
        "layer.1.attention.wq":  rng.normal(0, 0.09, (512, 512)),
        "layer.1.attention.wk":  rng.normal(0, 0.09, (512, 512)),
        "layer.1.mlp.w1":        rng.normal(0, 0.14, (2048, 512)),
        "layer.1.mlp.w2":        rng.normal(0, 0.11, (512, 2048)),
        "output":                rng.normal(0, 0.06, (512, 8000)),
        "norm.weight":           np.ones(512),
    }

    print(f"\nModel: {len(tensors)} layers")

    # Quick per-layer assessment
    print("\n--- Per-layer Assessment ---")
    print(f"{'Layer':30s} {'Sens':>6s} {'Level':>6s} {'Bits':>4s} {'MAE':>10s} {'Entropy':>8s}")
    print("-" * 70)

    engine = QuantEngine(target_compression=4.0)
    profiles = engine.profiler.profile_many(tensors)

    for p in profiles:
        level = engine.suggest_level(p)
        sens = engine._compute_sensitivity(p)
        # Simulate to get error estimate
        tensor = tensors[p.name]
        _, mae = simulate_quantize(tensor, level)
        print(f"{p.name:30s} {sens:6.4f} {level.name:>6s} {level.bits:4d} {mae:10.6f} {p.entropy:8.4f}")

    # Full strategy
    config = engine.strategize(profiles, original_bits=16)

    print(f"\n--- Quantization Strategy ---")
    print(f"Total params:  {config.total_params:>12,}")
    print(f"Original size: {config.original_size_bits / 8 / 1e9:.2f} GB (fp16)")
    print(f"Compressed:    {config.compressed_size_bits / 8 / 1e9:.2f} GB")
    print(f"Compression:   {config.overall_compression:.1f}x")
    print(f"Avg fidelity:  {1 - config.avg_fidelity_loss:.2%}")

    print(f"\n--- Hexagonal Groups ---")
    yao_names = ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]
    for s in config.strategies:
        yao = yao_names[s.hex_group]
        names = ", ".join(s.layer_names[:3])
        if len(s.layer_names) > 3:
            names += f" ... +{len(s.layer_names) - 3}"
        print(f"  {yao} [{s.level.name}] shared={s.shared_params} → {names}")


if __name__ == "__main__":
    main()
