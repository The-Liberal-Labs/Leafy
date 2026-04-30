# Phase 5: Synthesis — Cross-Paper Analysis & Gap Assessment

**Date**: 2026-05-01  
**Based on**: 70-paper database, 10 deep-dived papers, 10 open-source repos

## 1. Taxonomy of Multi-Agent Approaches for Leafy

```
Multi-Agent Enhancement of Plant Disease Classification
│
├── Agentic Training Orchestration (AutoML-Agent, Agent-R1)
│   ├── LLM Data Agent: dataset analysis, augmentation selection
│   ├── LLM Architecture Agent: backbone + head design
│   ├── LLM Training Agent: HPO, strategy selection
│   └── LLM Eval Agent: error analysis, report generation
│
├── Hierarchical Mixture of Experts (DirMixE, ExpertFlow, SoftMoE)
│   ├── Species Router: 27-way species classification
│   ├── Per-Species Disease Experts: 1-10 classes each
│   └── Soft/Hard Gating: differentiable vs discrete routing
│
├── Foundation Model Agents (Crop VQA, VLM for Vision)
│   ├── VLM Error Reviewer: explains misclassifications
│   ├── VLM Data Quality Agent: detects label noise
│   └── VLM Zero-shot Classifier: handles unseen diseases
│
├── Federated Multi-Farm Training (hivemind, FedNAS)
│   ├── Per-farm local training
│   ├── Privacy-preserving gradient aggregation
│   └── Non-IID client selection
│
└── Multi-Agent RL for Optimization (AOAD-MAT, MARL)
    ├── Parallel strategy exploration
    ├── Communication-based coordination
    └── Team formation for specialized exploration
```

## 2. Comparative Analysis of Approaches

### Approach A: Agentic Training Pipeline
| Aspect | Current Leafy | Proposed Agentic |
|--------|--------------|-----------------|
| Configuration | 40+ CLI args, hardcoded defaults | Natural language + structured config |
| Model selection | Manual architecture choice | Agent proposes based on dataset analysis |
| HPO | Fixed values from LR finder | Agent-driven dynamic HPO with RL |
| Error analysis | CSV export → manual review | VLM agent auto-reviews and explains |
| Modularity | 3072-line monolith | Agent-orchestrated micro-services |
| Implementation effort | — | HIGH (requires LLM API integration) |
| Risk | — | Agent decisions may be suboptimal without guardrails |

### Approach B: Hierarchical MoE Classifier
| Aspect | Current Leafy (Flat) | Proposed MoE |
|--------|---------------------|--------------|
| Classes | 90 flat | 27 species + 1-10 diseases each |
| Imbalance handling | ENS/Focal/Sampler | Natural — experts train on balanced subsets |
| Interpretability | Confusion matrix | Species misclassification vs disease confusion separated |
| Inference cost | 1 forward pass | Species router + 1 expert = 2 passes |
| Training cost | 1 model, 2 stages | 1 router + 27 experts (shared backbone) |
| Implementation effort | — | MEDIUM (model architecture changes) |
| Risk | — | Router errors cascade to wrong expert |

### Approach C: VLM Error Review Agent
| Aspect | Current Leafy | Proposed VLM Agent |
|--------|--------------|-------------------|
| Error analysis | Manual CSV review | Automated natural-language explanations |
| Data quality | Manual image check | VLM detects label noise automatically |
| Report generation | sklearn classification_report | Narrative error pattern report |
| Cost | 0 (human time) | API cost per image reviewed |
| Implementation effort | — | LOW (wrap existing outputs) |
| Risk | — | VLM may hallucinate explanations |

### Approach D: Federated Multi-Farm Training
| Aspect | Current Leafy | Proposed Federated |
|--------|--------------|-------------------|
| Data source | Single dataset | Multiple farm datasets |
| Privacy | N/A (public data) | Differential privacy for sensitive data |
| Data diversity | Fixed 27 species | Expandable per contributing farm |
| Training complexity | Single-GPU | Multi-node coordination |
| Implementation effort | — | HIGH (infrastructure + privacy) |
| Risk | — | Non-IID data degrades aggregation |

## 3. Recommended Implementation Priority

| Priority | Approach | Effort | Impact | Timeline |
|----------|----------|--------|--------|----------|
| **P0** | VLM Error Review Agent | Low | High | 1 week |
| **P1** | Hierarchical MoE Classifier | Medium | High | 2-4 weeks |
| **P2** | Active Learning Data Pipeline | Medium | Medium | 2 weeks |
| **P3** | Agentic Training Pipeline | High | High | 4-8 weeks |
| **P4** | Federated Multi-Farm Training | High | Medium | 8+ weeks |

### P0: VLM Error Review (Immediate)
- Wrap existing error review CSV → VLM prompt → structured output
- Use free/open models (LLaVA, Qwen-VL) to avoid API costs
- Integrate into `save_error_review_exports()` in train_efficientnet.py

### P1: Hierarchical MoE (Next)
- Add species classification head alongside disease head
- Implement soft-routed MoE using lucidrains/soft-moe-pytorch pattern
- Train on existing data_split with Species___condition hierarchy
- Expected improvement: +3-5% balanced accuracy, better minority F1

### P2: Active Learning (After MoE)
- Uncertainty sampling using ensemble disagreement (MoE experts)
- Iteratively flag low-confidence images for re-labeling
- Reduce manual review burden

### P3: Agentic Pipeline (Medium-term)
- Start with LLM Configuration Agent (replaces hardcoded args)
- Extend to LLM Training Monitor (reads logs, suggests adjustments)
- Full AutoML agent only after proven value of simpler agents

### P4: Federated (Long-term)
- Requires partner farms/institutions
- Build on hivemind or Flower framework
- Differential privacy for production deployment
