# Phase 4: Code & Tools — Open-Source Ecosystem Survey

**Date**: 2026-05-01

## Repository Summary

| # | Repository | Stars | Language | Focus | Leafy Relevance |
|---|-----------|-------|----------|-------|-----------------|
| 1 | DeepSpeed (Microsoft) | 42.2k | Python | MoE training + distributed inference | MoE training for species experts |
| 2 | MoE-LLaVA (PKU-YuanGroup) | 2.3k | Python | MoE for vision-language models | Vision MoE architecture reference |
| 3 | hivemind (Learning@Home) | 2.4k | Python | Decentralized deep learning | Multi-farm distributed training |
| 4 | mixtral-offloading | 2.3k | Python | MoE model offloading | Edge deployment of MoE |
| 5 | lucidrains/mixture-of-experts | 859 | Python | Sparse MoE PyTorch impl | Reference implementation |
| 6 | Microsoft/Tutel | 988 | C/Python | Optimized MoE library | Production MoE deployment |
| 7 | lucidrains/soft-moe-pytorch | 345 | Python | Soft MoE for vision | Species-router MoE |
| 8 | lucidrains/st-moe-pytorch | 382 | Python | ST-MoE implementation | Advanced MoE routing |
| 9 | keras-mmoe (Drawbridge) | 735 | Python | Multi-gate MoE (KDD 2018) | Multi-task MoE pattern |
| 10 | makeMoE | 801 | Python | From-scratch MoE tutorial | Educational reference |

## Repository Details

### 1. DeepSpeed (42.2k stars) — Microsoft
**URL**: https://github.com/deepspeedai/DeepSpeed
**Relevance**: **HIGHEST**
- Full MoE training support with expert parallelism, ZeRO optimization
- DeepSpeed-MoE: train models with trillions of parameters
- Expert/data/pipeline parallelism out of the box
- **Leafy integration**: Use DeepSpeed-MoE to train a 27-expert species MoE on single/multi-GPU. Zero code changes to model — just config.

### 2. MoE-LLaVA (2.3k stars) — PKU-YuanGroup
**URL**: https://github.com/PKU-YuanGroup/MoE-LLaVA
**Relevance**: HIGH
- MoE applied to vision-language models
- Shows that MoE routing works well for visual features
- **Leafy integration**: Reference for how to route visual features through experts. Architecture: shared vision encoder → MoE router → multiple expert FFNs.

### 3. hivemind (2.4k stars) — Learning@Home
**URL**: https://github.com/learning-at-home/hivemind
**Relevance**: HIGH
- Decentralized training across volunteer computers
- DHT-based peer discovery, fault-tolerant averaging
- **Leafy integration**: Enable farmers/institutions to collaboratively train a plant disease model without sharing raw data. Each farm contributes gradients only.

### 4. lucidrains/soft-moe-pytorch (345 stars)
**URL**: https://github.com/lucidrains/soft-moe-pytorch
**Relevance**: HIGH
- Soft MoE: every token goes to every expert (weighted), fully differentiable
- Originally from Google Brain's vision team
- **Leafy integration**: Direct implementation for vision MoE. Replace gating classifier with soft-routed MoE. Better gradient flow than hard routing.

### 5. OptiLLM (3.5k stars)
**URL**: https://github.com/algorithmicsuperintelligence/optillm
**Relevance**: MEDIUM
- Optimizing inference proxy for LLMs with MoE, MCTS, chain-of-thought
- **Leafy integration**: Pattern for multi-agent reasoning during inference. Multiple "agent" prompts analyze the image and vote.

## Frameworks & Tools

### MoE Libraries
| Library | Best For | Language |
|---------|----------|----------|
| DeepSpeed-MoE | Production training | Python/C++ |
| Tutel (Microsoft) | Optimized MoE kernels | C/Python |
| lucidrains MoE series | Research/experimentation | Python |
| keras-mmoe | Multi-task MoE | Python/TF |

### Distributed Training
| Library | Best For | Language |
|---------|----------|----------|
| hivemind | Volunteer/edge computing | Python |
| DeepSpeed ZeRO | Multi-GPU training | Python/C++ |
| PyTorch DDP | Standard distributed | Python |
| Flower | Federated learning | Python |

### VLM/VQA for Plant Disease
| Library | Best For | Language |
|---------|----------|----------|
| LLaVA | General VQA | Python |
| Qwen-VL | Multilingual VQA | Python |
| BLIP-2 | Image captioning/VQA | Python |
| Swin Transformer | Vision backbone | Python |

## Key Gap: No MoE-based plant disease system exists

After searching GitHub topics "mixture-of-experts" (399 repos) and "plant-disease-classification" (41 repos), **zero repositories combine MoE with plant disease classification**. This is a significant research and implementation gap that Leafy fills.

## Implementation Notes for Leafy

### MoE Integration Steps (using DeepSpeed or lucidrains):
1. Add species prediction head to existing EfficientNet backbone
2. Route image features to per-species disease classification experts
3. Train experts on balanced per-species subsets
4. Use soft gating for differentiable routing

### Distributed Training (using hivemind):
1. Wrap model in hivemind's `DecentralizedAverager`
2. Each GPU/farm trains on local data
3. Periodic gradient averaging via DHT
4. Optional differential privacy for sensitive farm data

### VLM Error Review (using LLaVA or Qwen-VL):
1. Export Leafy's misclassified images as PNG
2. Feed to VLM with prompt: "This image was classified as X by an AI model but the true label is Y. Explain why."
3. Aggregate VLM explanations into error pattern report
