from transformers import GPT2Config

class GPT2SparseConfig(GPT2Config):
    """
    GPT2 Config extended to include UltraEfficientSparseFFN hyperparameters.
    """
    model_type = "gpt2_sparse"

    def __init__(
        self,
        # GPT-2 original params
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        # Sparse FFN params
        ffn_k_freq=128,
        poly_degree=3,
        poly_keep_ratio=0.5,
        micro_steps=2,
        micro_keep_ratio=0.25,
        ffn_dropout=0.0,
        use_spectral=True,
        use_polynomial=True,
        use_micro=True,
        residual_gate_init=1.0,
        **kwargs
    ):
        super().__init__(vocab_size=vocab_size,
                         n_positions=n_positions,
                         n_ctx=n_ctx,
                         n_embd=n_embd,
                         n_layer=n_layer,
                         n_head=n_head,
                         **kwargs)
        # Sparse FFN hyperparameters
        self.ffn_k_freq = ffn_k_freq
        self.poly_degree = poly_degree
        self.poly_keep_ratio = poly_keep_ratio
        self.micro_steps = micro_steps
        self.micro_keep_ratio = micro_keep_ratio
        self.ffn_dropout = ffn_dropout
        self.use_spectral = use_spectral
        self.use_polynomial = use_polynomial
        self.use_micro = use_micro
        self.residual_gate_init = residual_gate_init