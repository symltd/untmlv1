import torch
from transformers import GPT2LMHeadModel, GPT2Config
from models.sparse_ffn import UltraEfficientSparseFFN

class GPT2Sparse(GPT2LMHeadModel):
    """
    GPT-2 model with UltraEfficientSparseFFN replacing standard MLP/FFN layers.
    """

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.patch_sparse_ffn()

    def patch_sparse_ffn(self):
        """
        Replace all transformer block MLPs with UltraEfficientSparseFFN.
        """
        hidden_size = self.config.n_embd
        for block in self.transformer.h:
            block.mlp = UltraEfficientSparseFFN(hidden_size)

    @classmethod
    def from_pretrained_sparse(cls, model_name_or_path: str, **kwargs):
        """
        Load pretrained GPT-2 weights and patch with Sparse FFN.
        """
        config = GPT2Config.from_pretrained(model_name_or_path, **kwargs)
        model = cls(config)
        # Load pretrained weights (except MLPs, which are replaced)
        pretrained_model = GPT2LMHeadModel.from_pretrained(model_name_or_path, **kwargs)
        model.transformer.load_state_dict(
            {k: v for k, v in pretrained_model.transformer.state_dict().items() if "mlp" not in k},
            strict=False
        )
        return model
