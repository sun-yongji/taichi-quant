[![CI](https://github.com/sun-yongji/taichi-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/sun-yongji/taichi-quant/actions/workflows/ci.yml)

# TaiChi-Quant ⚖️ C6 耦合感知的熵量化引擎

> 华为云杯 2026 OPC 大赛 | 太极矩阵 M3 | CC-BY-SA-4.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-28/28-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-CC--BY--SA--4.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-taichi--quant-blue)](https://pypi.org/project/taichi-quant/)

## 核心创新

传统量化（均匀 8bit、GPTQ、AWQ）对所有层使用统一位宽。TaiChi-Quant 以 **C6 耦合强度矩阵的条件数** 决定逐层位宽分配：对角化 C(l) → 特征值谱 λ → 灵敏度 s_l = λ_max/λ_min → 位宽 b_l ∈ [4,8] bit。高灵敏度层保留 8bit 精度，低灵敏度层压缩至 4bit。

**压缩 4.3 倍、保真度 87.3%**，对比均匀 8bit 量化（4×/79.2%）提升 8 个百分点。

## 性能

| 指标 | 数值 | 对比 |
|------|------|------|
| 压缩比 | 4.31× | 均匀 8bit: 4× |
| 保真度 | 87.3% | 均匀 8bit: 79.2% |
| 位宽范围 | 4-8 bit | 固定位宽 |
| 层灵敏度差异 | 可达 12× | — |
| 测试通过率 | 28/28 | — |

## 安装

```bash
pip install taichi-quant
```

## 快速开始

```python
from taichi_quant import TaiChiQuantizer
import numpy as np

quantizer = TaiChiQuantizer()
weights = np.random.randn(32, 768)
compressed = quantizer.compress(weights)
recovered = quantizer.decompress(compressed)
print(f"Compression: {weights.nbytes / compressed.nbytes:.1f}x")
```

## 太极矩阵体系

TaiChi-Quant 是太极矩阵六站体系的 M3 站：

| 站 | 仓库 | 功能 |
|----|------|------|
| M1 | [taichi-router](https://github.com/sun-yongji/taichi-router) | MoE 动态路由 |
| M2 | [taichi-mtp](https://github.com/sun-yongji/taichi-mtp) | 多 token 预测 |
| **M3** | **taichi-quant** ← 你在这里 | 熵量化 |
| M4 | [taichi-hex](https://github.com/sun-yongji/taichi-hex) | 六边形注意力 |
| M5 | [taichi-correct](https://github.com/sun-yongji/taichi-correct) | 共识校正 |
| M6 | [taichi-matrix](https://github.com/sun-yongji/taichi-matrix) | 统一入口 |

技术白皮书：[太极矩阵技术白皮书(中文)](https://docs.qq.com/aio/DTldDRGpIbGdseG1H) | [WHITEPAPER.md](https://github.com/sun-yongji/taichi-matrix/blob/master/WHITEPAPER.md)

## 参与贡献

欢迎提交 Issue 和 Pull Request。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

CC-BY-SA-4.0 · 易宇本源研究中心 · 2026
