# Model Experimentation Branch

This branch is for testing alternative summarization models to compare against the current LED-base production model.

## Available Models

Run `python -m src.models.model_configs` to see all 9 available models:

| Model | Context Length | GPU Memory | Speed | Quality | Domain |
|-------|----------------|------------|-------|---------|--------|
| **led-base** (current) | 16K tokens | 6 GB | Slow | High | Scientific |
| **led-large** | 16K tokens | 12 GB | Very Slow | Very High | Scientific |
| **longt5-base** | 16K tokens | 5 GB | Medium | High | General |
| **longt5-large** | 16K tokens | 10 GB | Slow | Very High | General |
| **distilbart** | 1K tokens | 2 GB | Fast | Medium | News |
| **bart-large-cnn** | 1K tokens | 4 GB | Medium | High | News |
| **pegasus-large** | 1K tokens | 5 GB | Medium | High | News |
| **flan-t5-base** | 512 tokens | 3 GB | Fast | Medium | General |
| **flan-t5-large** | 512 tokens | 6 GB | Medium | High | General |

## Quick Start

### 1. Run a Quick Comparison
Compare 3 models on a sample paper:

```bash
python experiments/compare_models.py --arxiv 1706.03762 --models led-base longt5-base distilbart
```

### 2. Test a Single New Model
Try Long-T5 (Google's alternative to LED):

```bash
python experiments/compare_models.py --arxiv 2103.14899 --models longt5-base
```

### 3. Compare All Long-Context Models
Test all models that support 16K+ tokens:

```bash
python experiments/compare_models.py --arxiv 1706.03762 --models led-base led-large longt5-base longt5-large
```

### 4. Speed Test (Fast Models Only)
Compare lightweight models:

```bash
python experiments/compare_models.py --arxiv 1706.03762 --models distilbart flan-t5-base
```

## Output

The script generates:
- **Console output**: Real-time progress, ROUGE scores, timing
- **JSON file**: Detailed results saved to `experiment_results.json`

### Sample Output:
```
⏱️  SPEED RANKING (fastest to slowest):
   1. distilbart: 12.34s
   2. longt5-base: 28.56s
   3. led-base: 45.78s

🏆 QUALITY RANKING (by ROUGE-2 F1):
   1. led-base: 0.3456
   2. longt5-base: 0.3123
   3. distilbart: 0.2789
```

## Recommended Experiments

### Experiment 1: Find the Sweet Spot
**Goal**: Balance speed vs quality  
**Command**:
```bash
python experiments/compare_models.py --arxiv 1706.03762 --models led-base longt5-base bart-large-cnn distilbart
```

### Experiment 2: Long Document Performance
**Goal**: Test on a very long paper (50+ pages)  
**Command**:
```bash
python experiments/compare_models.py --arxiv 2005.14165 --models led-base led-large longt5-large
```

### Experiment 3: Domain Specificity
**Goal**: Does scientific pre-training matter?  
**Test on biomedical paper**:
```bash
python experiments/compare_models.py --arxiv 2101.00234 --models led-base longt5-base
```

### Experiment 4: Resource-Constrained Testing
**Goal**: Best model for CPU-only or low GPU memory  
**Command**:
```bash
python experiments/compare_models.py --arxiv 1706.03762 --models distilbart flan-t5-base
```

## Interpreting Results

### ROUGE Scores (Higher = Better)
- **ROUGE-1**: Unigram overlap (0.30-0.45 is typical for abstractive)
- **ROUGE-2**: Bigram overlap (0.10-0.25 is good)
- **ROUGE-L**: Longest common subsequence (0.25-0.40 is typical)

### What to Look For:
1. **ROUGE-2 > 0.20** = Good quality summary
2. **Speed < 30s** = Acceptable for production
3. **GPU Memory ≤ 6GB** = Works on most research GPUs

## Expected Findings

Based on literature, we expect:

| Model | Expected ROUGE-2 | Expected Speed |
|-------|------------------|----------------|
| led-base | 0.18-0.25 | 40-60s |
| longt5-base | 0.16-0.23 | 25-40s |
| distilbart | 0.12-0.18 | 10-20s |

**Hypothesis**: Long-T5 might offer better speed/quality trade-off than LED for general scientific papers.

## Next Steps

After experiments, decide:
1. **Keep LED-base** if it has best ROUGE scores
2. **Switch to Long-T5** if it's faster with similar quality
3. **Offer both** as user options (fast vs quality mode)
4. **Add model auto-selection** based on paper length

## Merging Back

If you find a better model:
```bash
# Update abstractive.py with new default
git add src/models/abstractive.py
git commit -m "Switch default model to longt5-base (15% faster, similar ROUGE)"

# Merge back to main
git checkout main
git merge experiment/alternative-models
```

## Notes

- First run will download models (~1-3GB each)
- Models are cached in `~/.cache/huggingface/`
- GPU memory estimates assume float32; use fp16 for 50% reduction
