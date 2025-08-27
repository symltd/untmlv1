# gpt2_sparse_config.py
from transformers import GPT2Config
from typing import List, Optional

class GPT2SparseConfig(GPT2Config):
    """
    GPT-2 configuration extended to support:
    - Dynamic UltraEfficientSparseFFN sparsity per layer
    - Optional per-layer FFN expansion control
    """

    def __init__(
        self,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        ffn_expansion: int = 4,
        ffn_sparsity: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__(n_embd=n_embd, n_layer=n_layer, n_head=n_head, **kwargs)
        self.ffn_expansion = ffn_expansion

        # If sparsity list is provided, use per-layer sparsity; else uniform
        if ffn_sparsity is not None:
            assert len(ffn_sparsity) == n_layer, "Length of ffn_sparsity must equal n_layer"
            self.ffn_sparsity = ffn_sparsity
        else:
            self.ffn_sparsity = [0.5] * n_layer  # default 50% sparsity

    def get_ffn_config_for_layer(self, layer_idx: int):
        """
        Returns a dict of FFN parameters for a given layer:
        - hidden_size (d_model)
        - expansion_factor
        - sparsity
        """
        return {
            "hidden_size": self.n_embd,
            "expansion_factor": self.ffn_expansion,
            "sparsity": self.ffn_sparsity[layer_idx]
        }
