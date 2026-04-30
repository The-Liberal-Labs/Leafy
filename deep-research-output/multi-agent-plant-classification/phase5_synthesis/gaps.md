# Phase 5: Research Gaps — What Doesn't Exist Yet

## Gap 1: No MoE-based plant disease classification system
**Severity**: HIGH — Opportunity  
After searching 399 MoE repos and 41 plant disease repos, **zero** combine both. Leafy can be the first open-source hierarchical MoE for plant disease classification.

## Gap 2: No agentic AutoML for agricultural computer vision
**Severity**: HIGH — Opportunity  
AutoML-Agent (2024) targets tabular data only. No multi-agent LLM framework exists for vision training pipelines, especially agricultural ones.

## Gap 3: No VLM fine-tuned for plant pathology error analysis
**Severity**: MEDIUM  
Crop VQA (2026) exists for question-answering but no VLM is specifically trained to review classification errors and suggest label corrections. General VLMs (GPT-4V, LLaVA) haven't been evaluated on this task.

## Gap 4: No federated plant disease benchmark
**Severity**: MEDIUM  
Federated learning papers exist for general agriculture but no standard benchmark simulates multi-farm non-IID plant disease data distribution. Needed for evaluating federated Leafy.

## Gap 5: No soft-routed MoE for fine-grained classification
**Severity**: LOW  
Soft MoE (Google Brain) targets general vision. Hasn't been applied to fine-grained tasks like plant disease where expert specialization per species would be natural.

## Gap 6: No multi-agent RL for training pipeline optimization
**Severity**: LOW  
MARL is well-studied for game playing and robotics but not for ML pipeline optimization. Agent-R1 (2025) is the closest, using end-to-end RL for LLM agent training but not for vision model training.

## Gap 7: No ensemble diversity metric for plant disease experts
**Severity**: LOW  
Existing ensemble diversity metrics (KL-divergence, disagreement) haven't been calibrated for the plant disease domain where "similar-looking" diseases (e.g., different fungal spots) are semantically distinct.

## Summary

| Gap | Opportunity Level | Time to Fill |
|-----|-------------------|-------------|
| MoE plant disease classification | **Publishable novelty** | 2-4 weeks |
| Agentic AutoML for agri-vision | **Publishable novelty** | 4-8 weeks |
| VLM error review for plant pathology | Strong contribution | 1 week |
| Federated plant disease benchmark | Strong contribution | 4+ weeks |
| SoftMoE for fine-grained classification | Incremental | 2 weeks |
| MARL for training optimization | Exploratory | 8+ weeks |
