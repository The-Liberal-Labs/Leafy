# Phase 3: Deep Dive — Detailed Paper Analysis

**Papers read in detail**: 4 (Crop Disease VQA, ExpertFlow, DirMixE, partial AutoML-Agent)  
**Additional abstract-level notes**: 6

---

## Paper 1: Crop Disease VQA — Two-Stage Vision-Language Framework (2026)
**arXiv**: 2601.05143 | **Authors**: Ansary, Hossain et al.

### Problem
Plant disease diagnosis requires both visual identification AND explanatory reasoning. Existing systems output only labels without context, limiting real-world utility. Farmers need to ask "What disease is this?" and "What should I do?"

### Methodology
- **Stage 1**: Train Swin-T backbone for plant + disease classification (multitask, 10 epochs, AdamW, lr=1e-4)
- **Stage 2**: Freeze vision encoder, train BART/T5 decoder for VQA (2-3 epochs, lr=2e-5)
- **Dataset**: CDDM — 16 crops, 60 diseases, 1M+ QA pairs, 130K images
- **Architecture**: Swin-T → projection adapter → BART/T5 decoder
- **Explainability**: Grad-CAM for visual attention, token-level attribution for linguistic relevance

### Key Results
| Model | Plant Acc | Disease Acc | Params | Inference |
|-------|-----------|-------------|--------|-----------|
| Swin-T5 | 99.94% | 99.06% | 251M | 373ms |
| Swin-BART | 99.92% | 97.30% | 167.5M | 206ms |
| Qwen-VL-Chat | 97.4% | 91.5% | 7B | 12s |
| LLaVA-AG | 98.0% | 91.8% | 7B | 9.1s |

### Leafy Takeaway
1. **Two-stage training is validated**: pretrain vision encoder FIRST, then freeze for downstream. Leafy already does this.
2. **VQA capability**: Leafy could add a VQA decoder after training to enable farmers to ask questions about diagnosis.
3. **Lightweight wins**: Swin-BART (167M) beats 7B models — Leafy's EfficientNet-V2-S (21M) could match with similar two-stage training.
4. **Explainability**: Grad-CAM integration into Leafy's error review pipeline would add trust.

---

## Paper 2: ExpertFlow — Efficient MoE Inference (2024)
**arXiv**: 2410.17954 | **Authors**: He, Zhang, Wang et al.

### Problem
Sparse MoE models face memory challenges during inference. Existing offloading uses rigid LRU caching that doesn't adapt to dynamic expert routing. GPU memory for MoE (e.g., Mixtral 8x7B needs 96GB) exceeds single-GPU capacity.

### Methodology — Three Components:
1. **Routing Path Predictor**: T5-based transformer predicts which experts each token needs BEFORE computation. Reformulates routing as classification (active/inactive per expert). Trained independently from MoE — no fine-tuning needed.
2. **Expert Cache Engine (ECE)**: Predictive Locality-aware Expert Caching (PLEC) uses predicted routing to prefetch experts. Real-time correction swaps mispredicted experts during computation.
3. **Token Scheduler**: K-means clustering groups tokens with similar routing paths into same batch. Reduces number of experts activated per batch, improves cache hit ratio.

### Key Results
- GPU memory savings: 74.3% average, 93.72% peak
- Speedup: 2-10x over SE-MoE baseline
- Cache hit ratio: up to 91.96%, improves LRU by 27.65% average
- Predictor accuracy: 75-87% batch-level for Switch-32/64/128

### Leafy Takeaway
1. **Species-Router MoE analogy**: If Leafy uses a 27-expert species-based MoE, ExpertFlow's predictive caching would enable deployment on resource-constrained edge devices.
2. **Token Scheduler → Image Scheduler**: Group test images by predicted species to reduce expert switching. Similar K-means approach on image embeddings.
3. **Predictive loading**: Pre-load the right species expert before inference based on a lightweight species classifier. The species router IS the predictor.

---

## Paper 3: DirMixE — Hierarchical Label Distribution for Long-Tail (2024)
**arXiv**: 2405.07780 | **Authors**: Yang, Xu, Wang et al. (ICML)

### Problem
Test-agnostic long-tail recognition: test label distribution is unknown and can vary arbitrarily (forward LT, uniform, backward LT). Existing MoE methods (SADE, BalPoE) use FIXED test distributions per expert — they capture global variations but miss LOCAL variations.

### Methodology
1. **Dirichlet Mixture Meta-Distribution**: Model test label distribution as sampled from mixture of Dirichlet distributions. Each component captures LOCAL variations around a mean (forward, uniform, backward). Global diversity captured by differences between components.
2. **Monte Carlo Sampling**: Sample M test distributions from mixture. Each expert assigned to one component.
3. **Semi-Variance Regularization**: Replace full variance penalty with semi-variance (only penalize losses ABOVE mean). Prevents over-regularization of easy distributions.
4. **Theoretical Guarantee**: Sharper generalization bound due to variance regularization. O(N^(-1/2) + M^(-1/2)) improved to O(N^(-1/2) + M^(-3/4)).

### Key Results (CIFAR-100-LT, ImageNet-LT, iNaturalist)
| Method | Mean Acc | Std Dev | Best on Backward-LT |
|--------|----------|---------|---------------------|
| SADE | 60.71 | ±6.86 | 53.82 |
| DirMixE | 61.50 | ±6.43 | 56.25 |
| RIDE | 54.49 | ±8.47 | 43.49 |

### Leafy Takeaway — CRITICAL
1. **Leafy's `Species___condition` hierarchy maps directly to DirMixE**: Each species is a "component" — within each species, diseases have Dirichlet-distributed class frequencies. Cross-species imbalance is the "global variation."
2. **Replace flat 90-class classifier with hierarchical MoE**: Species-router (27-way) → per-species disease experts (1-10 classes). Train each expert with DirMixE-style loss.
3. **Semi-variance regularization**: Instead of just macro F1 (which penalizes variance equally), use semi-macro-F1 that only penalizes below-average per-class F1 scores.
4. **Test-time adaptation**: Self-supervised weight assignment for expert averaging — the model learns to weight species experts based on test distribution.

---

## Paper 4: AutoML-Agent — Multi-Agent LLM Framework (2024) (partial)
**arXiv**: 2410.02958 | **Authors**: Trirat, Jeong, Hwang

### Key Concepts (from abstract + survey reading)
- **Architecture**: Planner Agent coordinates specialized agents (Data, Model, Eval, Deploy)
- Each agent = LLM with domain-specific prompts and tools
- Full pipeline automation: data preprocessing → feature engineering → model selection → HPO → evaluation → deployment
- SOTA on 20+ tabular datasets

### Leafy Takeaway
1. **Replace monolithic trainer with agent-orchestrated pipeline**: Data Agent analyzes class distribution → Architecture Agent selects backbone → Training Agent runs with dynamic HPO → Eval Agent interprets results
2. **Natural language configuration**: "Train EfficientNet-V2-S with focal loss, macro F1 selection, 300px images, ENS beta=0.999" → agent translates to config
3. **Error review agent**: VLM agent examines classification errors and generates natural language explanations

---

## Papers 5-10: Brief Notes (abstract-level + partial reading)

### Paper 5: Hierarchical Pre-Training of Vision Encoders with LLMs (2026)
**arXiv**: 2604.00086
- LLM guides visual feature learning through hierarchical objectives
- **Leafy application**: Pre-train Leafy backbone with plant-biology-aware features from agricultural LLM

### Paper 6: Communication-Efficient Decentralized Multi-Agent Learning (2024)
**arXiv**: 2405.00839
- Balanced training workload across heterogeneous agents
- **Leafy application**: Distributed training across GPUs with varying VRAM

### Paper 7: Complementary Subspace LoRA for Few-Shot VLM (2025)
**arXiv**: 2501.15040
- Efficient VLM adaptation for few-shot classes using low-rank subspaces
- **Leafy application**: Fine-tune VLM on low-support classes (205-image Watermelon healthy) for zero-shot augmentation

### Paper 8: Warmup-Distill for Knowledge Distillation (2025)
**arXiv**: 2502.11766
- Gradual warmup bridges teacher-student distribution gap during KD
- **Leafy application**: Distill EfficientNet-V2-S teacher (21M) into MobileNet-V3 student (5.5M), preserving minority class performance

### Paper 9: Generalizing Teacher Networks for KD (2024)
**arXiv**: 2407.16040
- Teacher generalization across arbitrary student architectures
- **Leafy application**: One EfficientNet teacher → distill to MobileNet, ConvNeXt, etc.

### Paper 10: Agent-R1 — Training LLM Agents with End-to-End RL (2025)
**arXiv**: 2511.14460
- End-to-end RL training of LLM agents without supervised fine-tuning
- **Leafy application**: RL-based training agent that learns optimal hyperparameters through trial and error

---

## Cross-Paper Insights

1. **Two-stage training is universally validated**: Crop VQA, DirMixE, ExpertFlow all use two-stage approaches — pretrain then specialize. Leafy already does this correctly.

2. **MoE > single model for imbalance**: DirMixE proves MoE beats single-model across all test distributions. ExpertFlow proves MoE can be efficient. Leafy should adopt species-based MoE.

3. **Predictive scheduling is key to efficiency**: ExpertFlow's routing predictor shows that predicting which expert/submodel is needed BEFORE computation eliminates I/O overhead. The same principle applies to Leafy's inference: predict species first, then route to specialist.

4. **Hierarchy matters**: DirMixE exploits label hierarchy for better generalization. Leafy's `Species___condition` naming is a natural hierarchy that is currently unused.

5. **VLM for error analysis is proven**: Crop VQA achieves 99%+ accuracy with explainability. Leafy can add a VLM error-review agent without modifying the classifier.
