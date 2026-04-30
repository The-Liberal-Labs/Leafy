# Phase 2: Survey — Multi-Agent Approaches for Plant Disease Classification

**Date**: 2026-05-01  
**Papers**: 70 curated from 237 (filtered by relevance, recency ≥2020)  
**Median year**: 2023, 91 papers from 2024+

## Landscape Overview

The intersection of multi-agent systems and plant disease classification spans 10 thematic clusters:

| Theme | Papers | Core Idea | Leafy Applicability |
|-------|--------|-----------|---------------------|
| Multi-Agent ML & AutoML | 27 | LLM agents orchestrate ML pipelines | Replace monolithic trainer |
| VLM & Foundation Models | 20 | Vision-language models for zero-shot & error analysis | Quality reviewer agent |
| Agent RL & Coordination | 16 | MARL for distributed training optimization | Parallel strategy exploration |
| Class Imbalance & Long-Tail | 15 | Specialized loss/sampling for rare classes | Improve minority class F1 |
| Knowledge Distillation | 13 | Teacher-student compression for deployment | Edge model deployment |
| Mixture of Experts | 11 | Expert specialization per sub-task | Species-specific classifiers |
| Federated & Distributed | 8 | Multi-site collaborative training | Multi-farm data pooling |
| HPO & NAS | 8 | Automated architecture/hyperparameter search | Replace hardcoded config |
| Active Learning & Data Quality | 6 | Uncertainty-based data curation | Clean labeling errors |
| Plant Disease & Agriculture AI | 4 | Domain-specific vision AI for crops | Benchmark and SOTA |

---

## Cluster 1: Multi-Agent ML & AutoML (27 papers) — HIGHEST PRIORITY

This cluster represents the most direct upgrade path for Leafy.

### Key Papers:

**AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML** (Trirat et al., 2024) `arXiv:2410.02958`
- Architecture: Planner Agent coordinates specialized agents (Data, Model, Evaluation, Deployment)
- Each agent is an LLM with task-specific prompts and tools
- Achieves state-of-the-art on 20+ tabular datasets
- **Leafy application**: Replace `train_efficientnet.py` with an agent that reads data distribution, proposes architectures, tunes hyperparameters, and evaluates — all via natural language or structured config.

**POLARIS: Typed Planning and Governed Execution for Agentic AI** (2026) `arXiv:2601.11816`
- Typed planning with formal execution guarantees
- **Leafy application**: Ensures training pipeline correctness — no more manual "did I set the right batch size?"

**Agent-R1: Training Powerful LLM Agents with End-to-End RL** (2025) `arXiv:2511.14460`
- End-to-end RL training of LLM agents
- **Leafy application**: An agent that learns to optimize training hyperparameters through trial-and-error RL across multiple runs.

**Don't Just Demo, Teach Me the Principles** (2025) `arXiv:2502.07165`
- Principle-based multi-agent prompting for classification
- **Leafy application**: Multiple classification agents with different principles (one focused on minority classes, one on overall accuracy) voting on predictions.

### Multi-Agent Architecture Patterns

1. **Hierarchical**: One Planner + N Specialists (AutoML-Agent pattern)
2. **Cooperative**: Peer agents with communication (MARL pattern)  
3. **Ensemble Voting**: Independent agents with weighted voting (classification pattern)
4. **Debate**: Agents argue predictions, reach consensus (language models pattern)

---

## Cluster 2: VLM & Foundation Models (20 papers)

### Key Papers:

**A Two-Stage Multitask Vision-Language Framework for Explainable Crop Disease VQA** (2026) `arXiv:2601.05143`
- Directly applicable: VLM for crop disease with explanations
- Two-stage: disease detection → explainable QA
- **Leafy application**: Use as error-review agent — "Why did you predict Cassava Mosaic instead of Cassava Brown Streak?"

**Hierarchical Pre-Training of Vision Encoders with LLMs** (2026) `arXiv:2604.00086`
- LLM-guided visual feature learning through hierarchical objectives
- **Leafy application**: Pre-train Leafy's EfficientNet backbone with plant-biology-aware features from LLM knowledge.

**Complementary Subspace Low-Rank Adaptation of VLMs for Few-Shot Classification** (2025) `arXiv:2501.15040`
- Efficient VLM fine-tuning for few-shot classes
- **Leafy application**: Fine-tune a VLM on low-support classes (e.g., Watermelon healthy = 205 images) for zero-shot augmentation.

**FADE: Few-shot/zero-shot Anomaly Detection Engine using Large VLM** (2024) `arXiv:2409.00556`
- VLM-based anomaly detection without training
- **Leafy application**: Detect mislabeled or out-of-distribution plant images before training.

---

## Cluster 3: Mixture of Experts & Ensemble (11 papers)

### Key Papers:

**Generalizing GNNs with Tokenized Mixture of Experts** (2026) `arXiv:2602.09258`
- Tokenized routing for expert specialization
- **Leafy application**: Route leaf images to species-specific experts via a learned tokenizer.

**Convergence Rates for Softmax Gating Mixture of Experts** (2025) `arXiv:2503.03213`
- Theoretical guarantees for MoE convergence under different routing strategies
- **Leafy application**: Guarantees that the species-router MoE architecture will converge.

**ExpertFlow: Efficient MoE Inference via Predictive Expert Caching** (2024) `arXiv:2410.17954`
- Caches expert predictions to reduce inference cost
- **Leafy application**: Deploy 27-expert MoE efficiently on mobile/edge devices.

**DirMixE: Hierarchical Label Variations for Long-Tail Recognition** (2024) `arXiv:2405.07780`
- Hierarchical label structure to handle imbalance
- **Leafy application**: `Species___condition` naming already provides hierarchy — exploit it.

### MoE Architecture for Leafy:
```
Input Image → Species Router (27-way)
               ├── Apple Expert (4 diseases)
               ├── Cassava Expert (5 diseases)
               ├── Tomato Expert (10 diseases)
               └── ... 24 more species experts
```

Benefits:
- Each expert only handles 1-10 classes (vs 90 flat)
- Natural class imbalance handling — experts train on balanced subsets
- Interpretable errors — if router misclassifies species, it's a different failure mode
- Modular updates — add a new plant species without retraining all experts

---

## Cluster 4: Federated & Distributed Learning (8 papers)

### Key Papers:

**Communication-Efficient Training Workload Balancing for Decentralized Multi-Agent Learning** (2024) `arXiv:2405.00839`
- Balances training load across heterogeneous agents
- **Leafy application**: Distribute training across GPUs with different VRAM capacities.

**Distribution-Controlled Client Selection for Federated Learning** (2025) `arXiv:2509.20877`
- Smart client selection for non-IID data
- **Leafy application**: Select which farms contribute to training based on their crop diversity.

**Towards Non-I.I.D. Federated Deep Learning via NAS** (2020) `arXiv:2004.08546`
- NAS in federated settings for heterogeneous data
- **Leafy application**: Auto-search architectures per farm's crop profile.

---

## Cluster 5: Class Imbalance & Active Learning (21 papers combined)

### Key Papers:

**DirMixE: Harnessing Test Agnostic Long-tail Recognition** (2024)
- Hierarchical label variations — directly applicable to `Species___condition` hierarchy.

**Few-shot Adaptation of Medical Vision-Language Models** (2024) `arXiv:2409.03868`
- VLM adaptation with minimal data
- **Leafy application**: Zero-shot classification for low-support classes using VLM knowledge.

**Graph Embedded Intuitionistic Fuzzy RVFL Network for Class Imbalance** (2023)
- Graph-based imbalance handling with fuzzy logic
- **Leafy application**: Model relationships between similar plant diseases (e.g., different fungal infections look similar).

---

## Cluster 6: Knowledge Distillation (13 papers)

### Key Papers:

**Warmup-Distill: Bridge Distribution Mismatch Before KD** (2025) `arXiv:2502.11766`
- Addresses teacher-student distribution gap during distillation
- **Leafy application**: Distill the 27-expert MoE into a single mobile-friendly model.

**Generalizing Teacher Networks for Effective KD Across Student Architectures** (2024) `arXiv:2407.16040`
- Teacher generalization for arbitrary student architectures
- **Leafy application**: Distill EfficientNet-V2-S teacher into MobileNet-V3 student for edge deployment.

---

## Research Gap Analysis

| Gap | Current State | Multi-Agent Solution |
|-----|--------------|---------------------|
| Monolithic training script | 3072-line single file | Agent-orchestrated modular pipeline |
| Flat 90-class classifier | One-head classification | Hierarchical MoE (27 routers + N experts) |
| Hardcoded hyperparameters | Fixed in code | Agent-driven HPO with RL |
| Manual error review | CSV export for human review | VLM agent auto-reviews errors |
| Single-GPU training | No distributed support | Multi-agent distributed training |
| Static data pipeline | One-time split | Active learning agent for iterative improvement |
| No cross-farm collaboration | Single dataset | Federated multi-farm training |
