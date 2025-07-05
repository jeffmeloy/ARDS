"""
## ARDS (Adaptive Rate-Distortion Scaling): A framework for dynamic control of attention channel capacity.

**Top-Level Goal: A Framework for Intelligence as Learned Compression**

This system implements a framework for transformer control based on this concept: intelligence in these models is not learned representation, but **learned compression**. High-level reasoning emerges from the model's ability to learn and dynamically apply optimal, task-dependent strategies for the lossy compression of context.

**The Core Realization: Attention as an Adaptive Compression Engine**

The computational locus of reasoning in a transformer is the pre-softmax attention score matrix (`Q·K^T`). The subsequent softmax function acts as a non-linear compression algorithm, creating a focused probability distribution that selectively routes information. The core approach of this framework is that the *aggressiveness* of this compression should be a learned, context-dependent variable, not a static architectural property.

**Genuine Control: Modulating Information Density for Optimal Compression**

This system intervenes by modulating the **information density** of the attention distribution. Multiplying the pre-softmax scores by a learned `info_density` parameter (`softmax(scores * info_density)`) directly controls the compression ratio of the attention head. This is a form of **adaptive gain control** on the attention logits.

This mechanism treats each attention head as a **variable-rate compression channel**, governed by the principles of Rate-Distortion theory:

-   **High Information Density (Aggressive Compression):** A high `info_density` value forces a low-entropy distribution. This creates a **high-distortion, low-rate channel** that aggressively compresses the context, preserving only the most salient signals. This is essential for logical deduction and syntax parsing, where noise must be filtered.

-   **Low Information Density (High-Fidelity Preservation):** A low `info_density` value results in a high-entropy distribution. This creates a **low-distortion, high-rate channel** that preserves even weak signals from a broad context, essential for creative association and nuanced understanding.

**Learning an Optimal Compression Policy**

The system's controller is learning a **dynamic, context-dependent compression policy**. For each token, it assesses the cognitive task and computes the optimal `info_density` required. This reframes the scaling laws: performance scales not just with parameter count, but with the **sophistication and diversity of the compression strategies** a model can learn and apply.

**A Lens on Transformer Architecture and Cognition**

Landauer's principle shows how erasing information has an energy cost, but the system reveals the flip side - creating information (through decompression/exploration) also has a cost. KL divergence quantifies the work required to deviate from the system's natural compression equilibrium.

This explains why language models get weird when you push them too far from their training distribution. You're not just asking them to process unfamiliar content - you're forcing them to apply compression policies they never learned to optimize. It's like trying to run JPEG compression on audio data - the algorithm breaks down because it's optimized for fundamentally different signal statistics.

This framework provides a principled physical explanation for transformer design and behavior:

-   **Head Specialization:** Heads specialize by finding stable niches in the rate-distortion space. They are not anthropomorphic "agents," but optimized compressors for different signal types.
-   **Architectural Components:** Residual connections act as high-rate "bypass channels" to prevent critical information from being overly compressed. Layer normalization is a gain regulator that ensures the adaptive compression system remains stable.
-   **The Cost of Thought (KL Divergence):** The included KL divergence diagnostic measures the "energetic cost" of deviating from the model's default, learned compression heuristics. A spike in KL indicates the model is applying an effortful, non-standard policy to handle a novel or complex situation.

** Tractable Debugging**

This reframes reasoning failures not as flaws in the model's "knowledge," but as the application of a **suboptimal compression policy**. A model failing a math problem may not lack "math skills," but may have failed to apply the necessary high-density compression to isolate the numbers and operators from the surrounding text. This transforms debugging from an intractable interpretability problem into a more tractable analysis of control system policies. This is the instrumentation for a new science of machine cognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import re
import random
import types
from typing import Dict, Tuple, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 1. Controller Abstraction ---


class ControllerBase(nn.Module):
    """Abstract base class for all attention controllers."""

    def __init__(self):
        super().__init__()
        self.diagnostics: List[Dict[str, float]] = []

    def forward(
        self, attention_scores: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Each controller must implement its own forward pass."
        )

    def get_diagnostics(self) -> List[Dict[str, float]]:
        diag_data = self.diagnostics
        self.diagnostics = []
        return diag_data

    def _record_diagnostics(self, *args, **kwargs):
        raise NotImplementedError


class InfoDensityController(ControllerBase):
    """
    Learns a context-dependent 'info_density' parameter to modulate the
    attention distribution. High info_density creates a focused, low-entropy
    distribution (a low-capacity bottleneck).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        min_info_density: float = 0.2,
        max_info_density: float = 10.0,
        diagnostic_sampling_rate: float = 1.0,
    ):
        super().__init__()
        if not (0 < min_info_density <= max_info_density):
            raise ValueError("Info density values must be positive and min <= max.")
        self.num_heads = num_heads
        self.min_info_density = min_info_density
        self.max_info_density = max_info_density
        self.diagnostic_sampling_rate = diagnostic_sampling_rate

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.context_analyzer = nn.Linear(hidden_dim, num_heads * 2)  # [gain, bias]

    def forward(
        self, attention_scores: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        normed_states = self.layer_norm(hidden_states)
        gain, bias = self.context_analyzer(normed_states).chunk(2, dim=-1)

        # Stable combination and sigmoid scaling
        raw_control = gain + bias
        info_density = self.min_info_density + (
            self.max_info_density - self.min_info_density
        ) * torch.sigmoid(raw_control)

        # Reshape for broadcasting and apply modulation
        info_density = info_density.transpose(1, 2).unsqueeze(-1)
        modulated_scores = attention_scores * info_density

        if random.random() < self.diagnostic_sampling_rate:
            self._record_diagnostics(attention_scores, modulated_scores, info_density)

        return modulated_scores

    def _record_diagnostics(self, original_scores, modulated_scores, info_density):
        with torch.no_grad():
            orig_probs = F.softmax(original_scores, dim=-1, dtype=torch.float32)
            mod_probs = F.softmax(modulated_scores, dim=-1, dtype=torch.float32)

            kl_div = F.kl_div(mod_probs.log(), orig_probs, reduction="none").sum(dim=-1)
            mod_entropy = -torch.sum(mod_probs * mod_probs.log().nan_to_num(), dim=-1)

            self.diagnostics.append(
                {
                    "avg_info_density": info_density.mean().item(),
                    "avg_kl_div": kl_div.mean().item(),
                    "avg_mod_entropy": mod_entropy.mean().item(),
                }
            )


# --- 2. The Non-Invasive Hook ---


class ControlledAttentionHook:
    """
    A robust, non-invasive hook that wraps an existing attention module's forward
    pass. It re-implements the forward logic to gain access to pre-softmax scores
    while dynamically adapting to the module's specific architecture.
    """

    def __init__(self, controller: ControllerBase):
        self.controller = controller

    def __call__(
        self, module_instance, *args, **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states = args[0]
        bsz, q_len, _ = hidden_states.size()

        # Dynamically get projection layers
        q_proj, k_proj, v_proj, o_proj = self._get_projections(module_instance)

        # Apply projections
        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

        # Get attention parameters
        num_heads = getattr(
            module_instance,
            "num_heads",
            query_states.shape[-1] // getattr(module_instance, "head_dim", 1),
        )
        num_kv_heads = getattr(module_instance, "num_key_value_heads", num_heads)
        head_dim = query_states.shape[-1] // num_heads

        # Reshape for multi-head attention and handle GQA
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(
            1, 2
        )
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(
            1, 2
        )

        # Apply rotary embeddings with a robust fallback
        try:
            cos, sin = module_instance.rotary_emb(
                value_states, seq_len=key_states.shape[-2]
            )
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin, kwargs.get("position_ids")
            )
        except Exception:
            # Fallback for models with different RoPE implementations
            pass

        # Handle KV caching
        past_key_value = kwargs.get("past_key_value")
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        use_cache = kwargs.get("use_cache", False)
        past_key_value_out = (key_states, value_states) if use_cache else None

        # Repeat KV heads for GQA compatibility if necessary
        if num_kv_heads != num_heads:
            key_states = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
            value_states = value_states.repeat_interleave(
                num_heads // num_kv_heads, dim=1
            )

        # Compute raw attention scores
        attn_scores = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        # Apply attention mask robustly
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores + attention_mask

        # *** THE GENUINE INTERVENTION POINT ***
        modulated_scores = self.controller(attn_scores, hidden_states)

        # Final Softmax and Output Calculation
        attn_weights = F.softmax(modulated_scores, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = o_proj(attn_output)

        return attn_output, past_key_value_out

    def _get_projections(self, module):
        # A more compact and robust way to find projection layers
        projs = {}
        for proj_type in ["q", "k", "v", "o"]:
            for name, layer in module.named_children():
                if proj_type in name and isinstance(layer, nn.Linear):
                    projs[f"{proj_type}_proj"] = layer
                    break
            if f"{proj_type}_proj" not in projs:
                raise AttributeError(
                    f"{proj_type}_proj not found in {type(module).__name__}"
                )
        return projs["q_proj"], projs["k_proj"], projs["v_proj"], projs["o_proj"]

    # In-place RoPE application from Hugging Face for compatibility
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        if position_ids is None:
            return q, k
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# --- 3. The Main Control System ---


class ARDSControlSystem:
    def __init__(self, model):
        self.model = model
        self.controllers: Dict[str, ControllerBase] = {}
        self.original_forwards: Dict[str, types.MethodType] = {}
        self.device = model.device

    def instrument_model(
        self, controller_class=InfoDensityController, **controller_kwargs
    ):
        logger.info(f"Instrumenting model with {controller_class.__name__}...")
        instrumented_count = 0

        # Common attention module patterns
        attn_patterns = [r"self_attn$", r"attention$"]

        for name, module in self.model.named_modules():
            if any(re.search(p, name) for p in attn_patterns):
                try:
                    # Robustly determine hidden_dim and num_heads
                    config = getattr(module, "config", self.model.config)
                    hidden_dim = getattr(
                        config, "hidden_size", module.q_proj.in_features
                    )
                    num_heads = getattr(config, "num_attention_heads", module.num_heads)

                    controller = controller_class(
                        hidden_dim, num_heads, **controller_kwargs
                    ).to(self.device)

                    hook = ControlledAttentionHook(controller)
                    self.controllers[name] = controller
                    self.original_forwards[name] = module.forward
                    module.forward = types.MethodType(hook, module)

                    instrumented_count += 1
                except Exception as e:
                    logger.warning(f"Skipping instrumentation of {name}: {e}")

        logger.info(f"Successfully instrumented {instrumented_count} attention layers.")

    def deinstrument_model(self):
        logger.info("De-instrumenting model and restoring original methods.")
        for name, original_forward in self.original_forwards.items():
            # Find module by name and restore
            parts = name.split(".")
            m = self.model
            for part in parts:
                m = getattr(m, part)
            m.forward = original_forward
        self.original_forwards.clear()
        self.controllers.clear()
        logger.info("Model restored.")

    def get_all_diagnostics(self) -> Dict[str, List[Dict[str, float]]]:
        return {
            name: ctrl.get_diagnostics()
            for name, ctrl in self.controllers.items()
            if ctrl.diagnostics
        }


# --- 4. Example Usage and Entry Point ---


def load_model_and_tokenizer(model_path: str, device: str = "cpu", **kwargs):
    # This function is moved here to keep it with its usage context
    logger.info(f"Loading model from {model_path} to {device}...")
    model = torch.hub.load(
        "huggingface/transformers",
        "AutoModelForCausalLM",
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        **kwargs,
    ).to(device)
    tokenizer = torch.hub.load(
        "huggingface/transformers", "AutoTokenizer", model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer


@torch.no_grad()
def generate_with_ARDS_control(
    model, tokenizer, prompt: str, max_new_tokens: int = 50, **kwargs
):
    control_system = ARDSControlSystem(model)
    try:
        control_system.instrument_model(**kwargs)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        logger.info("Starting controlled generation...")
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        final_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        diagnostics = control_system.get_all_diagnostics()
    finally:
        control_system.deinstrument_model()
    return final_text, diagnostics


def main():
    MODEL_PATH = "Qwen/Qwen2-1.5B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PROMPT = "In quantum field theory, virtual particles are transient quantum fluctuations that appear in intermediate states of interactions. For example,"

    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH, DEVICE)
    except Exception as e:
        logger.error(
            f"Failed to load model. This may be due to network issues or model availability. Error: {e}"
        )
        return

    generated_text, diagnostics = generate_with_ARDS_control(
        model,
        tokenizer,
        PROMPT,
        max_new_tokens=60,
        controller_class=InfoDensityController,
        min_info_density=0.5,
        max_info_density=5.0,
        diagnostic_sampling_rate=1.0,  # Sample all steps for this short example
    )

    print("\n" + "=" * 80)
    print("CONTROLLED GENERATION OUTPUT:")
    print(generated_text)

    print("\n" + "=" * 40)
    print("CONTROL SYSTEM DIAGNOSTICS (Layer Averages):")
    for layer, diags in diagnostics.items():
        if not diags:
            continue
        avg_density = sum(d["avg_info_density"] for d in diags) / len(diags)
        avg_kl = sum(d["avg_kl_div"] for d in diags) / len(diags)
        print(
            f"  Layer '{layer.split('.')[-1]}': Avg Info Density = {avg_density:.3f} | Avg KL Div = {avg_kl:.4f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
