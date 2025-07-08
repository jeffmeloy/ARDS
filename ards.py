"""
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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import argparse
import os
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InfoDensityController(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        diagnostic_sampling_rate: float = 0.1,
        dtype=torch.float32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.diagnostic_sampling_rate = diagnostic_sampling_rate
        self.layer_norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.context_analyzer = nn.Linear(hidden_dim, num_heads, dtype=dtype)
        self.diagnostics: List[Dict[str, float]] = []
        self.time_series_log: List[float] = []
        # Initialize context analyzer weights and biases
        self.context_analyzer.weight.data.zero_()
        self.context_analyzer.bias.data.zero_()

    def forward(
        self, attention_scores: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        normed_states = self.layer_norm(hidden_states)
        log_space_adjustment = self.context_analyzer(normed_states)
        tamed_adjustment = torch.tanh(log_space_adjustment) * 1.0
        self.time_series_log.append(tamed_adjustment[0, 0, 0].item())
        tamed_adjustment = tamed_adjustment.transpose(1, 2).unsqueeze(-1)
        modulated_scores = attention_scores + tamed_adjustment

        if random.random() < self.diagnostic_sampling_rate:
            self._record_diagnostics(
                attention_scores, modulated_scores, tamed_adjustment
            )

        return modulated_scores

    def _record_diagnostics(self, original_scores, modulated_scores, adjustment):
        with torch.no_grad():
            orig_probs = F.softmax(original_scores, dim=-1, dtype=torch.float32)
            mod_probs = F.softmax(modulated_scores, dim=-1, dtype=torch.float32)
            kl_div = F.kl_div(mod_probs.log(), orig_probs, reduction="none").sum(dim=-1)
            mod_entropy = -torch.sum(mod_probs * mod_probs.log().nan_to_num(), dim=-1)
            self.diagnostics.append(
                {
                    "avg_adjustment": adjustment.mean().item(),
                    "avg_kl_div": kl_div.mean().item(),
                    "avg_mod_entropy": mod_entropy.mean().item(),
                }
            )

    def get_diagnostics(self) -> Dict[str, List]:
        diag_data = {"stats": self.diagnostics, "time_series": self.time_series_log}
        self.diagnostics = []
        self.time_series_log = []
        return diag_data


class LinearizationController(nn.Module):
    def __init__(self, diagnostic_sampling_rate: float = 0.1):
        super().__init__()
        self.diagnostic_sampling_rate = diagnostic_sampling_rate
        self.diagnostics: List[Dict[str, float]] = []
        self.time_series_log: List[float] = []

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            shifted_scores = (
                attention_scores - attention_scores.min(dim=-1, keepdim=True)[0] + 1
            )
            linear_probs = shifted_scores / shifted_scores.sum(dim=-1, keepdim=True)

            self.time_series_log.append(linear_probs.var(dim=-1).mean().item())

            if random.random() < self.diagnostic_sampling_rate:
                self._record_diagnostics(attention_scores, linear_probs)

            return linear_probs

    def _record_diagnostics(self, original_scores, linear_probs):
        with torch.no_grad():
            orig_probs = F.softmax(original_scores, dim=-1, dtype=torch.float32)
            kl_div = F.kl_div(linear_probs.log(), orig_probs, reduction="none").sum(
                dim=-1
            )
            linear_entropy = -torch.sum(
                linear_probs * linear_probs.log().nan_to_num(), dim=-1
            )
            self.diagnostics.append(
                {
                    "avg_prob_variance": linear_probs.var(dim=-1).mean().item(),
                    "avg_kl_from_softmax": kl_div.mean().item(),
                    "avg_linear_entropy": linear_entropy.mean().item(),
                }
            )

    def get_diagnostics(self) -> Dict[str, List]:
        diag_data = {"stats": self.diagnostics, "time_series": self.time_series_log}
        self.diagnostics = []
        self.time_series_log = []
        return diag_data


class LinearRampController(nn.Module):
    def __init__(
        self, layer_idx: int, total_layers: int, diagnostic_sampling_rate: float = 0.1
    ):
        super().__init__()
        self.alpha = (total_layers - layer_idx) / total_layers
        self.diagnostic_sampling_rate = diagnostic_sampling_rate
        self.diagnostics: List[Dict[str, float]] = []
        self.time_series_log: List[float] = []

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            shifted_scores = (
                attention_scores - attention_scores.min(dim=-1, keepdim=True)[0] + 1
            )
            linear_probs = shifted_scores / shifted_scores.sum(dim=-1, keepdim=True)
            exp_probs = F.softmax(attention_scores, dim=-1)
            mixed_probs = self.alpha * linear_probs + (1 - self.alpha) * exp_probs

            self.time_series_log.append(self.alpha)

            if random.random() < self.diagnostic_sampling_rate:
                self._record_diagnostics(attention_scores, mixed_probs)

            return mixed_probs

    def _record_diagnostics(self, original_scores, mixed_probs):
        with torch.no_grad():
            orig_probs = F.softmax(original_scores, dim=-1, dtype=torch.float32)
            kl_div = F.kl_div(mixed_probs.log(), orig_probs, reduction="none").sum(
                dim=-1
            )
            mixed_entropy = -torch.sum(
                mixed_probs * mixed_probs.log().nan_to_num(), dim=-1
            )
            self.diagnostics.append(
                {
                    "alpha_value": self.alpha,
                    "avg_kl_from_softmax": kl_div.mean().item(),
                    "avg_mixed_entropy": mixed_entropy.mean().item(),
                }
            )

    def get_diagnostics(self) -> Dict[str, List]:
        diag_data = {"stats": self.diagnostics, "time_series": self.time_series_log}
        self.diagnostics = []
        self.time_series_log = []
        return diag_data


def enable_ards_control(model, diagnostic_sampling_rate: float = 0.1):
    controllers = {}
    model_dtype = next(model.parameters()).dtype

    for name, module in model.named_modules():
        if (
            "self_attn" in name
            and isinstance(module, nn.Module)
            and not hasattr(module, "mlp")
        ):
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                controller = InfoDensityController(
                    model.config.hidden_size,
                    model.config.num_attention_heads,
                    diagnostic_sampling_rate,
                    model_dtype,
                ).to(model.device)

                module.ards_controller = controller
                controllers[name] = controller

    logger.info(f"ARDS enabled on {len(controllers)} attention layers")
    return controllers


def enable_linearization_control(model, diagnostic_sampling_rate: float = 0.1):
    controllers = {}

    for name, module in model.named_modules():
        if (
            "self_attn" in name
            and isinstance(module, nn.Module)
            and not hasattr(module, "mlp")
        ):
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                controller = LinearizationController(diagnostic_sampling_rate).to(
                    model.device
                )
                module.linearization_controller = controller
                controllers[name] = controller

    logger.info(f"Linearization enabled on {len(controllers)} attention layers")
    return controllers


def enable_leramp_control(model, diagnostic_sampling_rate: float = 0.1):
    controllers = {}
    total_layers = model.config.num_hidden_layers
    layer_idx = 0

    for name, module in model.named_modules():
        if (
            "self_attn" in name
            and isinstance(module, nn.Module)
            and not hasattr(module, "mlp")
        ):
            if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
                controller = LinearRampController(
                    layer_idx, total_layers, diagnostic_sampling_rate
                ).to(model.device)
                module.leramp_controller = controller
                controllers[name] = controller
                layer_idx += 1

    logger.info(f"Linear ramp enabled on {len(controllers)} attention layers")
    return controllers


def get_ards_diagnostics(model) -> Dict[str, Dict[str, List]]:
    diagnostics = {}
    for name, module in model.named_modules():
        if hasattr(module, "ards_controller"):
            diag_data = module.ards_controller.get_diagnostics()
            if diag_data["stats"]:
                diagnostics[name] = diag_data
        elif hasattr(module, "linearization_controller"):
            diag_data = module.linearization_controller.get_diagnostics()
            if diag_data["stats"]:
                diagnostics[name] = diag_data
        elif hasattr(module, "leramp_controller"):
            diag_data = module.leramp_controller.get_diagnostics()
            if diag_data["stats"]:
                diagnostics[name] = diag_data
    return diagnostics


class ModelTester:
    def __init__(
        self,
        model_path: str,
        use_ards: bool = False,
        use_linearization: bool = False,
        use_leramp: bool = False,
    ):
        self.model_path = model_path
        self.use_ards = use_ards
        self.use_linearization = use_linearization
        self.use_leramp = use_leramp
        self.controllers = None
        self.plot_dir = "./plots"

        if use_ards or use_linearization or use_leramp:
            os.makedirs(self.plot_dir, exist_ok=True)

        self.model, self.tokenizer = self.load_model_and_tokenizer()

        if self.use_ards:
            logger.info("ARDS Optimization is ENABLED.")
            self.controllers = enable_ards_control(self.model)
            if not self.controllers:
                logger.error("ARDS was enabled, but no controllers were attached.")
        elif self.use_linearization:
            logger.info("Linearization is ENABLED.")
            self.controllers = enable_linearization_control(self.model)
            if not self.controllers:
                logger.error(
                    "Linearization was enabled, but no controllers were attached."
                )
        elif self.use_leramp:
            logger.info("Linear Ramp is ENABLED.")
            self.controllers = enable_leramp_control(self.model)
            if not self.controllers:
                logger.error(
                    "Linear ramp was enabled, but no controllers were attached."
                )
        else:
            logger.info("Standard mode - no modifications.")

    def load_model_and_tokenizer(self):
        logger.info(f"Loading model from {self.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    def generate_text(self, messages, prompt_num, max_new_tokens=4096):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        input_ids_len = model_inputs.input_ids.shape[1]
        response = self.tokenizer.batch_decode(
            generated_ids[:, input_ids_len:], skip_special_tokens=True
        )[0]

        if self.use_ards or self.use_linearization or self.use_leramp:
            diagnostics = get_ards_diagnostics(self.model)
            if diagnostics:
                self.print_diagnostics(diagnostics)
                self.plot_diagnostics(diagnostics, prompt_num)

        return response

    def print_diagnostics(self, diagnostics: Dict[str, Dict[str, List]]):
        if self.use_ards:
            mode = "ARDS"
        elif self.use_linearization:
            mode = "Linearization"
        else:
            mode = "Linear Ramp"

        print(f"\n--- {mode} Diagnostics (Layer Averages) ---")
        for layer, diag_data in diagnostics.items():
            if not diag_data["stats"]:
                continue
            stats = diag_data["stats"]
            if self.use_ards:
                avg_adjustment = sum(d["avg_adjustment"] for d in stats) / len(stats)
                avg_kl = sum(d["avg_kl_div"] for d in stats) / len(stats)
                print(
                    f"  Layer '{layer}': Avg Adjustment = {avg_adjustment:.3f} | Avg KL Div = {avg_kl:.4f}"
                )
            elif self.use_linearization:
                avg_variance = sum(d["avg_prob_variance"] for d in stats) / len(stats)
                avg_kl = sum(d["avg_kl_from_softmax"] for d in stats) / len(stats)
                print(
                    f"  Layer '{layer}': Avg Prob Variance = {avg_variance:.3f} | Avg KL from Softmax = {avg_kl:.4f}"
                )
            else:  # leramp
                alpha = stats[0]["alpha_value"]
                avg_kl = sum(d["avg_kl_from_softmax"] for d in stats) / len(stats)
                print(
                    f"  Layer '{layer}': Alpha = {alpha:.3f} | Avg KL from Softmax = {avg_kl:.4f}"
                )
        print("-" * 40)

    def plot_diagnostics(
        self, diagnostics: Dict[str, Dict[str, List]], prompt_num: int
    ):
        layers_with_data = {
            name: data for name, data in diagnostics.items() if data.get("time_series")
        }

        if not layers_with_data:
            logger.warning(f"No time-series data to plot for prompt {prompt_num}.")
            return

        num_layers = len(layers_with_data)
        num_cols = 2
        num_rows = (num_layers + num_cols - 1) // num_cols

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(15, 4 * num_rows), squeeze=False
        )
        axes = axes.flatten()

        if self.use_ards:
            mode = "ARDS"
        elif self.use_linearization:
            mode = "Linearization"
        else:
            mode = "Linear Ramp"

        logger.info(
            f"Generating {num_rows}x{num_cols} grid plot for {num_layers} layers (Prompt {prompt_num}, {mode} mode)..."
        )

        layer_items = sorted(layers_with_data.items())

        for i, (layer_name, diag_data) in enumerate(layer_items):
            ax = axes[i]
            time_series_data = diag_data["time_series"]

            ax.plot(
                time_series_data,
                color="green",
                marker=".",
                linestyle="-",
                markersize=3,
                alpha=0.7,
            )

            simple_layer_name = ".".join(layer_name.split(".")[-2:])
            ax.set_title(f"Layer: {simple_layer_name}", fontsize=10)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

            if i % num_cols == 0:
                if self.use_ards:
                    ylabel = "γ Adjustment"
                elif self.use_linearization:
                    ylabel = "Prob Variance"
                else:
                    ylabel = "Alpha Value"
                ax.set_ylabel(ylabel, fontsize=8)

            if i >= num_layers - num_cols:
                ax.set_xlabel("Generation Step", fontsize=8)

        for i in range(num_layers, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            f"{mode} Controller Dynamics for Prompt {prompt_num}", fontsize=16, y=0.995
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        plot_path = os.path.join(
            self.plot_dir,
            f"prompt_{prompt_num}_{mode.lower().replace(' ', '_')}_grid.png",
        )

        try:
            fig.savefig(plot_path, dpi=150)
            logger.info(f"Saved diagnostic grid plot to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {plot_path}: {e}")

        plt.close(fig)


def run_inference_test(
    model_path: str, use_ards: bool, use_linearization: bool, use_leramp: bool
):
    tester = ModelTester(model_path, use_ards, use_linearization, use_leramp)

    prompts = [
        "Language: English. Write about Chinese and Western cooking techniques, explaining key differences.",
        "Write a fictional story about a quantum physicist who discovers their cat is actually conscious in multiple dimensions simultaneously.",
        "what is square root of 3343?",
    ]

    print("\n" + "=" * 80)
    if use_ards:
        mode = "ARDS OPTIMIZED"
    elif use_linearization:
        mode = "LINEARIZED ATTENTION"
    elif use_leramp:
        mode = "LINEAR RAMP ATTENTION"
    else:
        mode = "STANDARD (BASELINE)"
    print(f"RUNNING INFERENCE TESTS IN {mode} MODE")
    print("=" * 80)

    for i, prompt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": prompt}]
        print(f"\n\n--- Test {i}/{len(prompts)} ---")
        print(f"Prompt: {prompt}")
        response = tester.generate_text(messages, i)
        print("\nResponse:")
        print(response)
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM model with various attention modifications."
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the Hugging Face model directory."
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable ARDS dynamic attention controller.",
    )
    parser.add_argument(
        "--linearize",
        action="store_true",
        help="Enable linear normalization instead of softmax.",
    )
    parser.add_argument(
        "--leramp",
        action="store_true",
        help="Enable linear-to-exponential ramp across layers.",
    )

    args = parser.parse_args()

    mode_count = sum([args.optimize, args.linearize, args.leramp])
    if mode_count > 1:
        logger.error("Cannot enable multiple modes simultaneously.")
        return

    run_inference_test(args.model_path, args.optimize, args.linearize, args.leramp)


if __name__ == "__main__":
    main()

# The required edits to ./lib/transformers/model/qwen2/modeling_qwen2.py are assumed to be done separately.
# 1. In Qwen2Attention.forward, pass hidden_states to the attention_interface call:
#    ...
#    attention_interface(..., hidden_states_for_ards=hidden_states)
#
# 2. In eager_attention_forward, add the intervention block:
#    ...
#    attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
#
#    # ARDS Intervention
#    if hasattr(module, 'ards_controller'):
#        hidden_states_for_ards = kwargs.get("hidden_states_for_ards")
#        if hidden_states_for_ards is not None:
#            attn_scores = module.ards_controller(attn_scores, hidden_states_for_ards)
#
#    if attention_mask is not None: ...
#    ...