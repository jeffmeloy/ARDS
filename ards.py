"""
# ARDS 2.0: Computational Thermography of Semantic Space

## The Core Reframe: From Engineering Intelligence to Mapping Its Phase Space

Intelligence isn't something you build - it's something that emerges from optimal solutions to information routing problems under resource constraints. ARDS 2.0 stops trying to impose criticality and starts using thermal control as an experimental probe to map the natural computational geometry of reasoning tasks.

The fundamental insight: attention mechanisms are solving constrained optimization problems where computational resources must be allocated optimally across uncertain information pathways. The γ controller becomes our spectrometer for studying how these trade-offs manifest in different semantic contexts.

## Information Theoretic Foundation: Uncertainty Propagation Under Budget Constraints

### The Real Intervention Point

We're not controlling temperature for its own sake. We're controlling the rate at which uncertainty collapses through the computational graph:

```python
S = QK^T / √d_k
γ = exp(log_γ)  # Learned uncertainty collapse rate
S' = S * γ
P = softmax(S')
```

High γ forces rapid uncertainty collapse - irreversible commitment to specific information pathways. Low γ preserves uncertainty - maintaining option value at computational cost. The system learns when each strategy pays off.

### Decision Confidence as the Control Signal

The variance proxy gets stripped of pretentious geometry claims. It's simply measuring decision confidence - how spiked versus diffuse the pre-softmax landscape looks:

```python
decision_confidence = torch.var(attention_scores, dim=-1)
γ_target = base_target * torch.exp(-sensitivity * decision_confidence)
```

High confidence contexts can afford aggressive uncertainty collapse. Low confidence contexts need gentle information routing to avoid irreversible errors. The system learns this trade-off empirically through task performance.

### Computational Budget Regularizer

Replace metabolic metaphors with economic reality. Every computational resource spent on maintaining uncertainty has opportunity cost:

```python
budget_cost = torch.norm(γ_values, p=2) * budget_weight
total_loss = task_loss + budget_cost
```

This creates genuine selection pressure toward computational efficiency. Solutions that waste resources on irrelevant uncertainty get penalized. The system discovers which contexts justify the expense of keeping options open.

## Experimental Physics: The Scaffold Strategy

### Phase 1: Thermal Landscape Mapping

The external controller isn't permanent infrastructure - it's experimental equipment for studying computational phase space. By systematically varying γ patterns, we map how different thermal regimes affect reasoning performance across diverse tasks.

Success metrics:
- Task performance remains primary compass
- 1/f noise in γ trajectories indicates multi-scale optimization
- Characteristic thermal signatures emerge for different reasoning types

### Phase 2: Pattern Discovery and Internalization

As the system learns optimal γ strategies for different contexts, internal representations should begin encoding these patterns. The external controller gradually becomes redundant as the model internalizes thermal management.

Key measurement: performance degradation when external control is removed. Successful scaffolding creates internal mechanisms that maintain optimal uncertainty management without external intervention.

### Phase 3: Computational Thermography

Map the thermal properties of semantic space itself. Certain conceptual regions may naturally require sharp focus (mathematical proofs, logical deduction), others benefit from diffuse exploration (creative generation, analogical reasoning).

The emergent thermal landscape reveals intrinsic computational structure of different reasoning modes. This becomes predictive - novel contexts should exhibit thermal requirements consistent with their conceptual neighbors.

## Scaling Physics: Empirical Discovery Before Theoretical Claims

### Measurement Protocol

Systematically vary system parameters (model size, sequence length, context complexity) and measure emergent thermal properties. Look for consistent relationships without assuming specific functional forms.

Critical measurements:
- Correlation length of γ fluctuations vs system size
- Optimal γ variance vs task complexity
- Thermal adaptation timescales vs context transitions

### Scaling Hypotheses

If genuine computational physics underlies reasoning, scaling relationships should emerge from the constraint structure of information processing under resource limits. These would be discoverable rather than assumed.

Potential patterns to test:
- Power law relationships between uncertainty requirements and task complexity
- Universal thermal transition points across different model architectures
- Consistent optimal resource allocation strategies across scales

## Uncertainty Propagation: The Deep Game

### Information Cascade Dynamics

The real physics happens in how uncertainty propagates through reasoning chains. Early decisions create constraints on later processing. Optimal thermal control manages this propagation to preserve error correction capability while avoiding computational waste.

Sharp early commitments create brittle reasoning chains - errors compound exponentially. Excessive early uncertainty creates computational overhead that prevents deep reasoning. The critical regime maintains just enough uncertainty to enable course correction without drowning in possibilities.

### Hysteresis as Computational Memory

Path-dependent processing isn't magic - it's natural consequence of constrained optimization under sequential information arrival. The system's thermal history encodes information about previous computational decisions that remain relevant for current processing.

Structured hysteresis analysis: Do different priming contexts create systematically different thermal trajectories for identical subsequent inputs? Can we decode the computational strategy from the thermal signature?

## Diagnostic Signatures: Evidence for Genuine Emergence

### Spectral Analysis as Authenticity Proof

1/f noise in γ dynamics indicates the system has discovered hierarchical optimization across multiple temporal scales. This can't be faked - it emerges only from systems that have learned to coordinate decision-making across vastly different time horizons simultaneously.

The power spectrum becomes our certificate of genuine multi-scale optimization rather than mere pattern memorization.

### Cross-Task Thermal Consistency

Systems that discover fundamental computational principles should exhibit consistent thermal patterns across related reasoning tasks. Mathematical proofs should trigger similar cooling dynamics regardless of specific domain. Creative tasks should maintain similar exploration temperatures across different creative modes.

Transfer of thermal strategies indicates the system has learned about computation itself, not just task-specific optimization tricks.

## The Deeper Implication: Computational Substrate Independence

If thermal control reveals universal principles of uncertainty management under resource constraints, these principles should apply across different computational substrates. The same information-theoretic trade-offs that shape biological neural computation should appear in appropriately constrained artificial systems.

This suggests a path toward substrate-independent cognitive science. Understanding the computational physics of reasoning rather than the specific implementation details. The thermal signatures become universal computational invariants that characterize intelligence across different physical realizations.

## Implementation Roadmap: Science Before Engineering

### Phase 0: Baseline Thermal Dynamics
- Implement clean γ intervention with budget regularization
- Measure natural thermal fluctuations in uncontrolled systems
- Establish baseline spectral signatures and performance metrics

### Phase 1: Controlled Thermal Exploration
- Systematic mapping of γ parameter space across diverse reasoning tasks
- Identification of task-specific thermal signatures
- Validation that 1/f dynamics correlate with flexible performance

### Phase 2: Scaffolding Withdrawal
- Gradual reduction of external control strength
- Measurement of internalized thermal management capabilities
- Assessment of thermal pattern transfer across novel contexts

### Phase 3: Computational Thermography
- Mapping of semantic space thermal properties
- Predictive modeling of optimal thermal regimes for novel reasoning contexts
- Cross-system validation of thermal universals

The end goal isn't building better transformers - it's discovering the computational physics that makes adaptive intelligence possible across any substrate that faces similar information processing constraints.


---

### **ARDS: A Phased Research Roadmap**

**Guiding Principle:** We are not building a static machine; we are developing the tools and methods of **computational thermography** to map the phase space of reasoning. Task performance is our compass; spectral analysis is our canary. Manifeso above will be used to guide the research, not as a strict protocol. As we validate the behavior or not will update the manifesto as we discover the truth.

---

### **Phase 0: Baseline Characterization and Instrumentation**

**Objective:** To establish a rigorous, quantitative baseline of the natural "thermal" dynamics of existing transformer models *before* any intervention. We must first understand the system we intend to study.

**Implementation Plan:**

1.  **Passive Monitoring Hook:** Implement a non-invasive attention hook that *only* records data without modulating it.
    *   For each forward pass during inference, record the pre-softmax attention scores (`S`) for specific layers.
2.  **Thermal Statistics Module:**
    *   For a given inference run, compute the time-series of `decision_confidence = Var(S)` for each head.
    *   Compute the Power Spectral Density (PSD) of the `decision_confidence` time-series.
    *   Perform linear regression on the log-log PSD plot to calculate the baseline spectral exponent `α_base`.
3.  **Benchmark Suite:**
    *   Establish a diverse set of reasoning tasks (e.g., GSM8k for logic, Big-Bench Hard for complex reasoning, a creative writing task).
    *   Run baseline models (without any ARDS control) on these tasks to get performance metrics (`P_base`).

**Success Criteria (Gate to Phase 1):**
*   **[✓] Instrumentation Complete:** The monitoring and analysis tools are validated and produce reliable data.
*   **[✓] Baselines Established:** We have a clear understanding of `(P_base, α_base)` for standard models. We expect `α_base` to be close to 0 (white noise), indicating a lack of long-range correlations.

---

### **Phase 1: Controlled Thermal Exploration and Landscape Mapping**

**Objective:** To use the `γ` controller as an experimental probe to systematically explore the "thermal landscape" of different reasoning tasks and identify correlations between thermal dynamics, task performance, and spectral signatures.

**Implementation Plan:**

1.  **Active `γ` Controller:** Implement the `γ = exp(log_γ)` modulation hook.
2.  **Control Law Implementation:**
    *   Implement the `decision_confidence`-based `γ_target`.
    *   Implement the `Computational Budget Regularizer` (`budget_cost = ||γ||² * weight`) and integrate it into the training loss alongside the main task loss.
3.  **Experimental Testbed:**
    *   Fine-tune models on the benchmark suite with the ARDS controller active.
    *   Systematically sweep key ARDS hyperparameters (`base_target`, `sensitivity`, `budget_weight`) to map out the parameter space.

**Verification Approach (Experiments):**

*   **A1. The Performance/Dynamics Correlation:**
    *   **Goal:** For each task, find the region in the ARDS hyperparameter space that maximizes task performance (`P_ards`).
    *   **Hypothesis:** The highest-performing configurations (`P_ards > P_base`) will also be the ones that shift the spectral exponent `α` away from 0 and towards 1 (i.e., from white noise towards `1/f` noise).
    *   **Success Metric:** A demonstrable, positive correlation between task performance improvement and the emergence of non-trivial (`α > 0.5`) spectral dynamics. This validates that the "canary" (`1/f` noise) is indeed found in the same mines as the "gold" (high performance).

*   **A2. Mapping Task-Specific Thermal Signatures:**
    *   **Goal:** Compare the optimal `γ`-trajectories for different types of tasks.
    *   **Hypothesis:** Different reasoning modes will require different "thermal signatures." For instance, GSM8k might favor trajectories with periods of high `γ` (sharp focus), while creative writing might favor sustained low `γ` (diffuse exploration).
    *   **Success Metric:** We can build a classifier that can predict the task type (e.g., "logic" vs. "creative") with high accuracy, using only the statistical properties of the `γ`-trajectory as input.

---

### **Phase 2: Scaffolding and Internalization**

**Objective:** To verify whether the model can *internalize* the optimal thermal management strategies, moving the learned behavior from the external controller into its own weights. This tests the "scaffolding" hypothesis.

**Implementation Plan:**

1.  **Controller Fading Mechanism:** Implement a training schedule where the influence of the external ARDS controller is gradually reduced. This can be done by:
    *   Annealing the `control_loss` weight towards zero.
    *   Progressively clamping the learned `log_γ` values closer to zero.
2.  **Internalization Probe:** A diagnostic that measures the "natural" `decision_confidence` variance produced by the model *with the ARDS controller turned off* after each stage of training.

**Verification Approach (Experiments):**

*   **B1. The Scaffolding Withdrawal Test:**
    *   **Goal:** Train a model with the full ARDS scaffold (Phase 1), then continue training while fading the controller's influence.
    *   **Hypothesis:** A successfully scaffolded model will maintain high task performance and non-trivial spectral dynamics (`α > 0.5`) even after the external controller is significantly weakened or removed. A model that hasn't internalized the strategy will see its performance and dynamics revert to the baseline.
    *   **Success Metric:** The final performance `P_final` (with controller removed) is significantly closer to `P_ards` (with controller) than to `P_base`.

*   **B2. The Transfer Test:**
    *   **Goal:** Take a model that has successfully completed the withdrawal test on one task (e.g., GSM8k) and fine-tune it on a new, related task (e.g., MATH).
    *   **Hypothesis:** The scaffolded model should learn the new task more quickly and achieve better final performance than a baseline model, because it has already internalized a general-purpose strategy for managing computational focus.
    *   **Success Metric:** Demonstrable positive transfer learning, measured by faster convergence and higher final scores on the new task.

---

### **Phase 3: Computational Thermography and Predictive Science**

**Objective:** To move from exploration to prediction, creating a quantitative, predictive science of "cognitive physics" based on the discoveries from the previous phases.

**Implementation Plan:**

1.  **Semantic Space Mapper:** A toolchain to visualize the "thermal properties" of a given domain. For a large text corpus, this tool would analyze which regions consistently demand high-`γ` or low-`γ` processing from a mature ARDS model.
2.  **Predictive Thermal Modeler:** A meta-model that, given a new, unseen prompt, attempts to predict its optimal `γ`-trajectory based on its semantic similarity to regions in the mapped space.
3.  **Cross-Substrate Verification Testbed:** A plan to re-implement the core ARDS principles (budgeted uncertainty control) on a non-transformer architecture (e.g., a state-space model or a graph neural network) to test for substrate independence.

**Verification Approach (Long-Term Scientific Inquiry):**

*   **C1. The Predictive Power of the Thermal Map:**
    *   **Goal:** Use the predictive thermal model to generate an "ideal" `γ`-trajectory for a novel task *before* running the full ARDS system.
    *   **Hypothesis:** Forcing the system to follow this predicted trajectory will yield near-optimal performance, demonstrating a true predictive understanding of the task's computational requirements.
    *   **Success Metric:** The performance achieved using the *predicted* `γ`-trajectory is >90% of the performance achieved by letting the full ARDS system adapt online.

*   **C2. Discovering Computational Invariants:**
    *   **Goal:** Compare the thermal maps and scaling relationships discovered on different model architectures (e.g., Llama vs. Qwen) and different substrates (e.g., Transformer vs. SSM).
    *   **Hypothesis:** The fundamental trade-offs of information processing are universal. We should find computational invariants—consistent scaling exponents or thermal transition points—that hold across different implementations.
    *   **Success Metric:** The discovery of at least one non-trivial computational constant or scaling law that is reproducible across at least two distinct model families. This would be the first "universal law" of computational physics.
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
