"""
## ARDS (Adaptive Rate-Distortion Scaling): A Framework for Dynamic Control of Attention Channel Capacity

**Top-Level Goal: Intelligence as Learned, Adaptive Compression**

This system implements a framework for transformer control based on a core concept: high-level
reasoning is not just learned representation, but **learned, adaptive compression**. The model's
intelligence emerges from its ability to dynamically apply optimal, context-dependent strategies
for the lossy compression of information.

**The Core Realization: Attention as a Variable-Rate Compression Channel**

The computational locus of reasoning is the pre-softmax attention score matrix. The subsequent
softmax function acts as a non-linear compression algorithm, selectively routing information.
The central thesis of ARDS is that the *aggressiveness* of this compression should not be a
static architectural property, but a learned, dynamic variable.

**Genuine Control: Unbounded, Multiplicative Scaling of Information Density**

This system intervenes by multiplying the pre-softmax scores by a learned `info_density`
parameter. This directly and powerfully controls the compression ratio of the attention head.
The control mechanism is designed to be as simple and unconstrained as possible:

- **Multiplicative Scaling:** The controller learns a scaling factor that multiplies the
  raw attention scores.
- **Neutral Baseline of 1.0:** The controller learns a value in log-space (`log_density`).
  This is converted to a scaler via `exp(log_density)`, making `1.0` the natural,
  "do-nothing" baseline. A learned output of `0` results in a scaler of `1.0`.
- **Unbounded Control:** There are no artificial hyperparameters like `min/max` ranges or `gain`
  factors. The controller has full authority to set any positive scaling factor. We trust the
  main training loop (e.g., through weight decay and gradient clipping) to provide emergent
  regularization, forcing the controller to learn sensible policies that improve performance.

This approach treats each attention head as a variable-rate compression channel, where the
`info_density` scaler dictates the Rate-Distortion trade-off:

- **High `info_density` (> 1.0):** Aggressive compression. Creates a high-distortion, low-rate
  channel essential for filtering noise and focusing on logical signals.
- **Low `info_density` (< 1.0):** High-fidelity preservation. Creates a low-distortion, high-rate
  channel essential for creative association and capturing broad context.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 1. Controller Abstraction and Implementation ---

class ControllerBase(nn.Module):
    """Abstract base class for all attention controllers."""
    def __init__(self):
        super().__init__()
        self.diagnostics: List[Dict[str, float]] = []

    def forward(self, attention_scores: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Each controller must implement its own forward pass.")

    def get_diagnostics(self) -> List[Dict[str, float]]:
        diag_data = self.diagnostics
        self.diagnostics = []
        return diag_data

    def _record_diagnostics(self, *args, **kwargs):
        raise NotImplementedError

class InfoDensityController(ControllerBase):
    """
    Learns a context-dependent 'info_density' scaling factor. This controller is
    unbounded, learning a direct log-space adjustment, allowing the training
    process to discover the optimal, unconstrained control policy.
    """
    def __init__(self, hidden_dim: int, num_heads: int, diagnostic_sampling_rate: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.diagnostic_sampling_rate = diagnostic_sampling_rate
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # The network directly outputs the desired log-space adjustment.
        self.context_analyzer = nn.Linear(hidden_dim, num_heads)

    def forward(self, attention_scores: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_states = self.layer_norm(hidden_states)
        log_space_adjustment = self.context_analyzer(normed_states)
        
        # Convert from log-space to a multiplicative scaler where exp(0) = 1 (no change).
        info_density_scaler = torch.exp(log_space_adjustment)
        
        # Reshape and apply the modulation.
        info_density_scaler = info_density_scaler.transpose(1, 2).unsqueeze(-1)
        modulated_scores = attention_scores * info_density_scaler

        if random.random() < self.diagnostic_sampling_rate:
            self._record_diagnostics(attention_scores, modulated_scores, info_density_scaler)
            
        return modulated_scores

    def _record_diagnostics(self, original_scores, modulated_scores, info_density_scaler):
        with torch.no_grad():
            orig_probs = F.softmax(original_scores, dim=-1, dtype=torch.float32)
            mod_probs = F.softmax(modulated_scores, dim=-1, dtype=torch.float32)

            kl_div = F.kl_div(mod_probs.log(), orig_probs, reduction='none').sum(dim=-1)
            mod_entropy = -torch.sum(mod_probs * mod_probs.log().nan_to_num(), dim=-1)

            self.diagnostics.append({
                "avg_info_density_scaler": info_density_scaler.mean().item(),
                "avg_kl_div": kl_div.mean().item(),
                "avg_mod_entropy": mod_entropy.mean().item(),
            })

# --- 2. The Non-Invasive Hook ---

class ControlledAttentionHook:
    """A robust hook that wraps an attention module's forward pass."""
    def __init__(self, controller: ControllerBase):
        self.controller = controller

    def __call__(self, module_instance, *args, **kwargs) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states = args[0]
        bsz, q_len, _ = hidden_states.size()

        q_proj, k_proj, v_proj, o_proj = self._get_projections(module_instance)

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

        config = getattr(module_instance, 'config', self.controller.model.config)
        num_heads = getattr(config, "num_attention_heads", module_instance.num_heads)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = query_states.shape[-1] // num_heads

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        
        try:
            cos, sin = module_instance.rotary_emb(value_states, seq_len=key_states.shape[-2])
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin, kwargs.get("position_ids"))
        except Exception:
            pass

        past_key_value = kwargs.get("past_key_value")
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value_out = (key_states, value_states) if kwargs.get("use_cache", False) else None

        if num_kv_heads != num_heads:
            key_states = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
            value_states = value_states.repeat_interleave(num_heads // num_kv_heads, dim=1)

        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores + attention_mask

        # *** INTERVENTION POINT ***
        modulated_scores = self.controller(attn_scores, hidden_states)

        attn_weights = F.softmax(modulated_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = o_proj(attn_output)

        return attn_output, past_key_value_out

    def _get_projections(self, module):
        projs = {}
        for proj_type in ['q', 'k', 'v', 'o']:
            # A simple heuristic to find the linear layers
            candidate = getattr(module, f"{proj_type}_proj", None)
            if isinstance(candidate, nn.Linear):
                projs[f"{proj_type}_proj"] = candidate
            else: # Fallback for different naming
                for name, layer in module.named_children():
                    if proj_type in name and isinstance(layer, nn.Linear):
                        projs[f"{proj_type}_proj"] = layer
                        break
            if f"{proj_type}_proj" not in projs:
                raise AttributeError(f"{proj_type}_proj not found in {type(module).__name__}")
        return projs['q_proj'], projs['k_proj'], projs['v_proj'], projs['o_proj']

    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        if position_ids is None: 
            return q, k
        cos, sin = cos[position_ids].unsqueeze(1), sin[position_ids].unsqueeze(1)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

# --- 3. The Main Control System ---

class ARDSControlSystem:
    def __init__(self, model):
        self.model = model
        self.controllers: Dict[str, ControllerBase] = {}
        self.original_forwards: Dict[str, types.MethodType] = {}
        self.device = model.device

    def instrument_model(self, controller_class=InfoDensityController, **controller_kwargs):
        logger.info(f"Instrumenting model with {controller_class.__name__}...")
        instrumented_count = 0
        attn_patterns = [r"self_attn$", r"attention$"]
        
        for name, module in self.model.named_modules():
            if any(re.search(p, name) for p in attn_patterns):
                try:
                    config = getattr(module, 'config', self.model.config)
                    hidden_dim = getattr(config, "hidden_size", next(p.in_features for p in module.parameters() if isinstance(p, nn.Linear)))
                    num_heads = getattr(config, "num_attention_heads", module.num_heads)
                    
                    # Pass the model reference to the hook constructor
                    hook = ControlledAttentionHook(
                        controller_class(hidden_dim, num_heads, **controller_kwargs).to(self.device)
                    )
                    hook.controller.model = self.model # Give controller access to model config if needed

                    self.controllers[name] = hook.controller
                    self.original_forwards[name] = module.forward
                    module.forward = types.MethodType(hook, module)
                    instrumented_count += 1
                except Exception as e:
                    logger.warning(f"Skipping instrumentation of {name}: {e}")
        
        logger.info(f"Successfully instrumented {instrumented_count} attention layers.")

    def deinstrument_model(self):
        logger.info("De-instrumenting model.")
        for name, original_forward in self.original_forwards.items():
            module = self.model.get_submodule(name)
            module.forward = original_forward
        self.original_forwards.clear()
        self.controllers.clear()
        logger.info("Model restored.")

    def get_all_diagnostics(self) -> Dict[str, List[Dict[str, float]]]:
        return {name: ctrl.get_diagnostics() for name, ctrl in self.controllers.items() if ctrl.diagnostics}

# --- 4. Example Usage and Entry Point ---

def load_model_and_tokenizer(model_path: str, device: str = "cpu", **kwargs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading model from {model_path} to {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

@torch.no_grad()
def generate_with_ARDS_control(model, tokenizer, prompt: str, max_new_tokens: int = 50, **kwargs):
    control_system = ARDSControlSystem(model)
    try:
        control_system.instrument_model(**kwargs)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, use_cache=True)
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
        logger.error(f"Failed to load model. This may be due to network issues or model availability. Error: {e}")
        return

    generated_text, diagnostics = generate_with_ARDS_control(
        model, tokenizer, PROMPT, max_new_tokens=60,
        controller_class=InfoDensityController,
        # No controller hyperparameters needed!
        diagnostic_sampling_rate=1.0,
    )

    print("\n" + "=" * 80)
    print("CONTROLLED GENERATION OUTPUT:")
    print(generated_text)

    print("\n" + "=" * 40)
    print("CONTROL SYSTEM DIAGNOSTICS (Layer Averages):")
    for layer, diags in diagnostics.items():
        if not diags: 
            continue
        avg_scaler = sum(d["avg_info_density_scaler"] for d in diags) / len(diags)
        avg_kl = sum(d["avg_kl_div"] for d in diags) / len(diags)
        print(f"  Layer '{layer.split('.')[-1]}': Avg Info Density Scaler = {avg_scaler:.3f} | Avg KL Div = {avg_kl:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
