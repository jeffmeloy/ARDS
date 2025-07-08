# ARDS: Attention Redistribution and Diagnostic System

## Overview

ARDS is a research framework for modifying and analyzing attention mechanisms in transformer models. It provides three experimental attention modification strategies along with comprehensive diagnostic tools to study how these changes affect model behavior during text generation.

## Core Capabilities

### Attention Modification Strategies

**1. InfoDensityController (`--optimize`)**
- **Purpose**: Learn context-dependent attention score adjustments
- **Method**: Uses a linear layer + LayerNorm to generate per-head adjustments based on hidden states
- **Implementation**: `adjustment = tanh(linear(layer_norm(hidden_states)))` added to pre-softmax scores
- **Use Case**: Testing whether models can learn better attention patterns through gradient descent

**2. LinearizationController (`--linearize`)**
- **Purpose**: Replace exponential softmax with linear normalization
- **Method**: `shifted_scores = scores - min(scores) + 1; probs = shifted_scores / sum(shifted_scores)`
- **Implementation**: Completely bypasses softmax computation
- **Use Case**: Testing whether linear attention is sufficient for language modeling

**3. LinearRampController (`--leramp`)**
- **Purpose**: Gradual transition from linear to exponential attention across layers
- **Method**: `mixed_probs = α * linear_probs + (1-α) * softmax_probs` where `α = (total_layers - current_layer) / total_layers`
- **Implementation**: Layer-dependent interpolation between attention schemes
- **Use Case**: Testing optimal attention "temperature" scheduling across model depth

### Diagnostic and Analysis Features

**Real-time Metrics Collection**
- KL divergence between original and modified attention distributions
- Attention entropy measurements (distribution sharpness/flatness)
- Variance tracking for attention probability distributions
- Configurable sampling rate to balance overhead vs. data collection

**Time Series Analysis**
- Per-layer controller parameter evolution during generation
- Statistical aggregation across attention heads
- Temporal pattern visualization with matplotlib grid plots
- Automatic plot generation and saving for offline analysis

**Performance Impact Assessment**
- Minimal computational overhead through sampling-based collection
- Non-invasive monitoring mode for baseline measurements
- Comparative analysis between modified and standard attention

## Critical Limitations and Requirements

### Qwen2/2.5 Model Compatibility Issues

**Transformers Library Modification Required**
This system requires direct access to pre-softmax attention scores and hidden states during the attention computation. Standard Hugging Face transformers do not expose these intermediate values, requiring manual modification of the model implementation.

**Required Code Changes in `transformers/models/qwen2/modeling_qwen2.py`:**

1. **Modify `Qwen2Attention.forward()` method:**
```python
# Add hidden_states parameter to attention_interface call
attention_interface(
    # ... existing parameters ...
    hidden_states_for_ards=hidden_states  # ADD THIS LINE
)
```

2. **Modify `eager_attention_forward()` function:**
```python
# After computing attention scores but before applying attention_mask
attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling

# ARDS INTERVENTION BLOCK - INSERT THIS
if hasattr(module, 'ards_controller'):
    hidden_states_for_ards = kwargs.get("hidden_states_for_ards")
    if hidden_states_for_ards is not None:
        attn_scores = module.ards_controller(attn_scores, hidden_states_for_ards)
elif hasattr(module, 'linearization_controller'):
    # Skip softmax entirely and return linear probabilities
    linear_probs = module.linearization_controller(attn_scores)
    # Continue with linear_probs instead of softmax(attn_scores)
elif hasattr(module, 'leramp_controller'):
    # Apply mixed linear/softmax normalization
    mixed_probs = module.leramp_controller(attn_scores)
    # Continue with mixed_probs instead of softmax(attn_scores)

# Continue with standard attention computation
if attention_mask is not None: ...
```

### Technical Constraints

**Model Architecture Dependencies**
- Currently designed for decoder-only transformer architectures
- Assumes standard Q/K/V attention structure with `self_attn` module naming
- Requires `q_proj` and `k_proj` attributes for module detection
- May need adaptation for models with different attention implementations

**Inference-Only Operation**
- Controllers are attached as `nn.Module` objects but are not integrated into training loops
- No gradient flow from task loss to controller parameters during fine-tuning
- Designed for experimental analysis rather than production training

**Memory and Performance Overhead**
- Diagnostic collection adds computational cost proportional to sampling rate
- Time series storage accumulates throughout generation (cleared after each prompt)
- Plot generation requires matplotlib and file I/O operations
- May impact generation speed, especially with high diagnostic sampling rates

## Usage and Installation

### Setup Requirements

1. **Install Dependencies**
```bash
pip install torch transformers matplotlib
```

2. **Modify Transformers Library**
Apply the code changes described above to your local transformers installation
```bash
# Locate your transformers installation
python -c "import transformers; print(transformers.__file__)"
# Edit: .../transformers/models/qwen2/modeling_qwen2.py
```

3. **Prepare Model**
Ensure you have a Qwen2/2.5 model available locally or via Hugging Face Hub

### Command Line Interface

```bash
# Baseline (standard attention)
python ards.py /path/to/qwen2-model

# Test InfoDensityController
python ards.py /path/to/qwen2-model --optimize

# Test LinearizationController  
python ards.py /path/to/qwen2-model --linearize

# Test LinearRampController
python ards.py /path/to/qwen2-model --leramp
```

### Output and Analysis

**Console Output**
- Real-time diagnostic statistics per layer
- Average adjustment values, KL divergences, entropy measurements
- Performance timing information

**Generated Files**
- `./plots/prompt_N_[mode]_grid.png`: Multi-layer attention dynamics visualization
- Time series plots showing controller parameter evolution during generation

## Research Applications

### Attention Mechanism Studies
- **Softmax Necessity**: Does exponential normalization provide significant benefits over linear?
- **Layer-wise Strategies**: How should attention "temperature" change across model depth?
- **Context Dependency**: Can attention patterns be improved through learned, context-aware adjustments?

### Model Behavior Analysis
- **Attention Entropy Evolution**: How does attention sharpness change during generation?
- **Distributional Stability**: How much do attention modifications affect output distributions?
- **Computational Trade-offs**: What's the performance cost of different attention schemes?

### Baseline Establishment
- **Standard Model Profiling**: Characterize attention patterns in unmodified models
- **Performance Benchmarking**: Establish baseline metrics for comparison studies
- **Pattern Documentation**: Create reference attention behavior for different prompt types

## Implementation Quality

**Strengths**
- Modular, extensible controller design
- Robust error handling and device management  
- Comprehensive diagnostic collection
- Production-ready CLI interface
- Non-invasive integration approach

**Limitations**
- Requires manual transformers library modification
- Inference-only operation (no training integration)
- Limited to Qwen2/2.5 architecture pattern
- Performance overhead from diagnostic collection

This framework provides a solid foundation for experimental attention mechanism research, with the primary barrier being the required transformers library modifications for accessing internal attention computations.
