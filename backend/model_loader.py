"""
Model Loader for LatentLab

Loads Qwen3-4B-Instruct-2507 (or other models) with hooks for activation extraction.
Designed for ROCm but works on CUDA too.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Callable, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default model - easy to swap
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class HookedModel:
    """
    Wrapper around a HuggingFace model with activation hooks.
    Allows extracting hidden states at any layer during forward pass.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        max_memory: Optional[Dict[int, str]] = None
    ):
        self.model_name = model_name
        self.dtype = dtype
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, 'hip') and torch.hip.is_available():
                self.device = "cuda"  # ROCm uses cuda API
            else:
                self.device = "cpu"
                logger.warning("No GPU detected, running on CPU (slow!)")
        else:
            self.device = device
            
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}, dtype: {dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device != "cpu" else None,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.device != "cpu" else "eager",
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
        # Get model config
        self.config = self.model.config
        self.n_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        
        logger.info(f"Model loaded: {self.n_layers} layers, hidden_size={self.hidden_size}")
        
        # Hook storage
        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        
    def _make_hook(self, name: str) -> Callable:
        """Create a hook function that stores activation"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                # Qwen returns (hidden_states, ...) tuple
                self._activations[name] = output[0].detach()
            else:
                self._activations[name] = output.detach()
        return hook
    
    def register_hooks(self, layers: Optional[List[int]] = None):
        """Register hooks on specified layers (or all if None)"""
        self.clear_hooks()
        
        if layers is None:
            layers = list(range(self.n_layers))
            
        # Qwen3 uses model.layers[i] structure
        for i in layers:
            layer = self.model.model.layers[i]
            name = f"layer_{i}"
            handle = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)
            
        logger.debug(f"Registered hooks on {len(layers)} layers")
        
    def clear_hooks(self):
        """Remove all hooks"""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._activations = {}
        
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations from last forward pass"""
        return self._activations.copy()
    
    def tokenize(self, text: str, return_tensors: str = "pt") -> Dict:
        """Tokenize input text"""
        return self.tokenizer(
            text, 
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
            max_length=2048
        )
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    @torch.no_grad()
    def forward_with_cache(
        self,
        text: str,
        layers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run forward pass and capture activations at specified layers.
        
        Returns:
            Dict with 'logits', 'activations', 'tokens'
        """
        self.register_hooks(layers)
        
        inputs = self.tokenize(text)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        result = {
            "logits": outputs.logits,
            "activations": self.get_activations(),
            "input_ids": inputs["input_ids"],
            "tokens": [self.tokenizer.decode([t]) for t in inputs["input_ids"][0]]
        }
        
        self.clear_hooks()
        return result
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate text from prompt"""
        inputs = self.tokenize(prompt)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        """
        Get embedding for text by taking mean of last layer hidden states.
        This is a proxy for dedicated embedding models.
        """
        result = self.forward_with_cache(text, layers=[self.n_layers - 1])
        last_layer = result["activations"][f"layer_{self.n_layers - 1}"]
        # Mean pool over sequence
        embedding = last_layer.mean(dim=1)
        return embedding.squeeze(0)
    
    def inject_at_layer(
        self,
        layer_idx: int,
        steering_vector: torch.Tensor,
        strength: float = 1.0
    ) -> Callable:
        """
        Create a hook that adds a steering vector at a specific layer.
        Used for Activation Addition (ActAdd) experiments.
        """
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                modified = hidden + (steering_vector.to(hidden.device) * strength)
                return (modified,) + output[1:]
            else:
                return output + (steering_vector.to(output.device) * strength)
        
        layer = self.model.model.layers[layer_idx]
        handle = layer.register_forward_hook(steering_hook)
        self._hooks.append(handle)
        return handle


def load_model(
    model_name: str = DEFAULT_MODEL,
    **kwargs
) -> HookedModel:
    """Convenience function to load a model"""
    return HookedModel(model_name=model_name, **kwargs)
