[![CI](https://github.com/sun-yongji/taichi-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/sun-yongji/taichi-quant/actions/workflows/ci.yml)

# TaiChi-Quant ⚖️ C6耦合感知的熵量化引擎

> 华为云杯2026 OPC大赛  |  太极矩阵 M3  |  Apache 2.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-28/28-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 核心创新

传统量化（均匀8bit、GPTQ、AWQ）对所有层使用统一位宽。TaiChi-Quant以**C6耦合强度矩阵的条件数**决定逐层位宽分配：对角化C(l)→特征值谱λ→灵敏度s_l=λ_max/λ_min→位宽b_l∈[4,8]bit。高灵敏度层保留8bit精度，低灵敏度层压缩至4bit。

**压缩4.3倍、保真度87.3%**，对比均匀8bit量化（4×/79.2%）提升8个百分点。

## 性能

| 指标 | 数值 | 对比 |
|------|------|------|
| 压缩比 | 4.31× | 均匀8bit: 4× |
| 保真度 | 87.3% | 均匀8bit: 79.2% |
| 位宽范围 | 4-8 bit | 固定位宽 |
| 层灵敏度差异 | 可达12× | — |
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

| 站 | 仓库 | 功能 |
|----|------|------|
| M1 | [taichi-router](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-router) | MoE动态路由 |
| M2 | [taichi-mtp](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-mtp) | 多token预测 |
| **M3** | **taichi-quant** ← 你在这里 | 熵量化 |
| M4 | [taichi-hex](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-hex) | 六边形注意力 |
| M5 | [taichi-correct](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-correct) | 共识校正 |
| M6 | [taichi-matrix](https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-matrix) | 统一入口 |

技术白皮书：[太极矩阵技术白皮书](https://docs.qq.com/aio/DTldDRGpIbGdseG1H)

## 许可

Apache 2.0 · 太极量子团队 · 2026