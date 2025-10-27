from .hf_wrapper import HuggingFaceWrapper


class DeepSpeedBackend(HuggingFaceWrapper):
    """
    Inference backend based on DeepSpeed
    """

    def __init__(self, *args, ds_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import deepspeed
        except ImportError as exc:
            raise ImportError(
                "Please install deepspeed to use DeepSpeedBackend: pip install deepspeed"
            ) from exc
        ds_config = ds_config or {"train_batch_size": 1, "fp16": {"enabled": True}}
        self.model, _, _, _ = deepspeed.initialize(model=self.model, config=ds_config)
        self.model.module.eval()
