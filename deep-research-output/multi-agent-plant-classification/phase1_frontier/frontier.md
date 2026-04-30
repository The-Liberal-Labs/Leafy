# Phase 1: Frontier Research — Multi-Agent Approaches for Plant Disease Classification

**Topic**: Multi-agent systems, ensemble methods, foundation models, and agentic AI applicable to improving the Leafy plant disease classification pipeline.

**Date**: 2026-05-01  
**Papers sourced**: 200 from arXiv (2020-2026), deduplicated to ~180 unique

## Key Trends & Breakthroughs

### 1. Agentic AutoML (Most Directly Applicable)
The convergence of LLM agents and AutoML is the most promising frontier for upgrading Leafy:

- **AutoML-Agent** (Trirat et al., 2024): A multi-agent LLM framework that automates the full ML pipeline — data preprocessing, feature engineering, model selection, hyperparameter tuning, and deployment. Uses specialized agents (Data Agent, Model Agent, Evaluation Agent) coordinated by a Planner Agent.
- **Agentic ML pipelines** replace manual scripting with LLM-delegated decisions: the agent reads the dataset, proposes architectures, tunes hyperparameters, and iteratively improves based on validation metrics.

**Implication for Leafy**: Replace the hardcoded `train_efficientnet.py` hyperparameters with an agent that dynamically selects training strategies based on class distribution analysis. Enable natural-language configuration ("train with focal loss, macro F1 selection, 300px images").

### 2. Mixture of Experts (MoE) for Fine-Grained Classification
MoE is rapidly advancing beyond LLMs into vision:

- **Tokenized MoE for GNNs** (2026): Shows how expert specialization improves fine-grained graph classification — directly transferable to per-species expert networks.
- **Softmax Gating MoE** (2025): Convergence guarantees for learned routing in MoE — applicable to routing leaf images to species-specific experts.
- **ExpertFlow** (2024): Efficient MoE inference via predictive expert caching — enables deploying multi-expert systems on edge devices.

**Implication for Leafy**: Replace the single 90-class classifier with a hierarchical MoE: a species-router (27-way) that delegates to per-species disease experts (1-10 classes each). This naturally handles inter-species imbalance.

### 3. Vision-Language Models (VLMs) for Agricultural AI
Foundation models are increasingly applied to plant science:

- **Hierarchical Pre-Training of Vision Encoders with LLMs** (2026): Shows LLM-guided visual feature learning — could improve plant disease feature extraction.
- **VoxelPrompt** (2024): A vision agent for end-to-end medical image analysis using LLM-guided prompting — directly transferable to plant pathology.
- **Vision-Language Pre-training surveys** (2022): Comprehensive reviews of VLP architectures applicable to plant-image-text alignment.

**Implication for Leafy**: Deploy a VLM (e.g., LLaVA, GPT-4V) as a "reviewer agent" that examines model errors, suggests mislabeled images, and provides natural-language explanations of failure modes.

### 4. Federated Learning for Distributed Agriculture
Multi-site collaborative training without data centralization:

- **Distribution-Controlled Client Selection** (2025): Smart client selection for federated learning in non-IID settings — critical for farms with different crop distributions.
- **Federated + Transfer Learning** (2022): Survey of defense mechanisms applicable to privacy-preserving plant disease data sharing across institutions.

**Implication for Leafy**: Enable multi-farm training where each farm contributes its local plant disease data without sharing raw images.

### 5. Active Learning & Data Quality Agents
- **Active Learning for Data Streams** (2023): Survey of uncertainty sampling methods — applicable to iterative data improvement in Leafy.
- **Data Pipeline Training with AutoML** (2024): Integrating AutoML to optimize data flow — directly addresses Leafy's data preparation pipeline.

### 6. Multi-Agent Reinforcement Learning for Training Optimization
- **AOAD-MAT** (2025): Multi-agent RL with agent ordering — applicable to coordinating multiple training strategies in parallel.
- **MARL with Communication surveys** (2022): Frameworks for agent coordination that could orchestrate distributed hyperparameter sweeps.

## Most Actionable Papers for Leafy (Top 15)

| # | Paper | Year | Relevance |
|---|-------|------|-----------|
| 1 | AutoML-Agent: Multi-Agent LLM Framework for Full-Pipeline AutoML | 2024 | Directly applicable — agent-based training orchestration |
| 2 | Data Pipeline Training: Integrating AutoML to Optimize Data Flow | 2024 | Data quality and preprocessing automation |
| 3 | VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis | 2024 | Vision agent architecture transferable to plant pathology |
| 4 | Hierarchical Pre-Training of Vision Encoders with LLMs | 2026 | LLM-guided visual feature learning for plants |
| 5 | ExpertFlow: Efficient Mixture-of-Experts Inference | 2024 | Efficient MoE deployment for edge devices |
| 6 | Convergence Rates for Softmax Gating Mixture of Experts | 2025 | Theoretical grounding for species-router MoE |
| 7 | Distribution-Controlled Client Selection for Federated Learning | 2025 | Multi-farm collaborative training |
| 8 | Task Discrepancy Maximization for Fine-grained Few-Shot Classification | 2022 | Handling rare disease classes |
| 9 | Vision-Language Pre-training: Basics, Recent Advances, Future Trends | 2022 | Foundation for VLM-based error analysis |
| 10 | Learn to Accumulate Evidence from All Training Samples | 2023 | Theoretical framework for ensemble evidence accumulation |
| 11 | Active Learning for Data Streams: A Survey | 2023 | Uncertainty-based data curation |
| 12 | Federated and Transfer Learning: Adversaries and Defense Mechanisms | 2022 | Security for distributed plant disease training |
| 13 | Generalizing GNNs with Tokenized Mixture of Experts | 2026 | MoE specialization patterns |
| 14 | VLP: A Survey on Vision-Language Pre-training | 2022 | VLM architectures for agricultural domain |
| 15 | Knowledge-Embedded Representation Learning for Fine-Grained Recognition | 2018 | External knowledge integration for plant disease features |

## Research Directions for Leafy Enhancement

### Direction A: Agentic Training Pipeline (HIGH priority)
Replace the 3072-line monolithic script with an LLM-agent-orchestrated pipeline:
- **Data Agent**: Analyzes class distribution, detects label noise, suggests augmentation
- **Architecture Agent**: Selects backbone and head design based on dataset characteristics
- **Training Agent**: Runs training with dynamic hyperparameter adjustment
- **Evaluation Agent**: Interprets results, generates reports, flags error patterns

### Direction B: Hierarchical MoE Classifier (HIGH priority)
Replace flat 90-class classifier with:
- **Stage 1 Router**: 27-way species classifier (EfficientNet-V2-S)
- **Stage 2 Experts**: Per-species disease classifiers (1-10 classes each)
- **Benefits**: Natural class imbalance handling, interpretable errors, modular updates

### Direction C: VLM Error Review Agent (MEDIUM priority)
Deploy a VLM (GPT-4V, LLaVA, or Qwen-VL) as post-training reviewer:
- Examines high-confidence misclassifications
- Suggests labeling errors with natural-language justification
- Generates per-class diagnostic reports

### Direction D: Federated Multi-Farm Training (MEDIUM priority)
Enable Leafy to train across geographically distributed farms:
- Each farm runs local training on its plant disease data
- Federated averaging aggregates model updates without sharing raw images
- Distribution-controlled client selection handles non-IID farm data

### Direction E: Multi-Agent Hyperparameter Optimization (LOW priority)
Coordinated agent-based HPO:
- Multiple agents explore different regions of the hyperparameter space
- Shared knowledge base of tried configurations
- Communication protocols for efficient exploration
