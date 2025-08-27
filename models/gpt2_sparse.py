from transformers import GPT2LMHeadModel
from models.sparse_ffn import UltraEfficientSparseFFN
from models.gpt2_sparse_config import GPT2SparseConfig

class GPT2Sparse(GPT2LMHeadModel):
    """
    GPT-2 model with UltraEfficientSparseFFN replacing standard MLPs.
    """
    config_class = GPT2SparseConfig

    def __init__(self, config: GPT2SparseConfig):
        super().__init__(config)
        self.patch_sparse_ffn()

    def patch_sparse_ffn(self):
        hidden_size = self.config.n_embd

        for block in self.transformer.h:
            block.mlp = UltraEfficientSparseFFN(
                hidden_dim=hidden_size,
                k_freq=self.config.ffn_k_freq,
                poly_degree=self.config.poly_degree,
                poly_keep_ratio=self.config.poly_keep_ratio,
                micro_steps=self.config.micro_steps,
                micro_keep_ratio=self.config.micro_keep_ratio,
                dropout=self.config.ffn_dropout,
                use_spectral=self.config.use_spectral,
                use_polynomial=self.config.use_polynomial,
                use_micro=self.config.use_micro,
                residual_gate_init=self.config.residual_gate_init
            )

    @classmethod
    def from_pretrained_sparse(cls, model_name_or_path: str, **kwargs):
        """
        Load pretrained GPT-2 weights and patch with Sparse FFN.
        """
        config = GPT2SparseConfig.from_pretrained(model_name_or_path, **kwargs)
        model = cls(config)
        # Load pretrained weights (except MLPs, replaced by sparse FFN)
        pretrained_model = GPT2LMHeadModel.from_pretrained(model_name_or_path, **kwargs)
        model.transformer.load_state_dict(
            {k: v for k, v in pretrained_model.transformer.state_dict().items() if "mlp" not in k},
            strict=False
        )
        return model
