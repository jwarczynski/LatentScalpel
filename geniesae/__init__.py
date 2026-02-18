"""GenIE SAE: Sparse Autoencoder pipeline for GENIE diffusion language model."""

from geniesae.sae import TopKSAE

# Lazy imports to avoid requiring pytorch-lightning/wandb at import time
def __getattr__(name):
    if name == "ActivationCollectionConfig":
        from geniesae.configs.collection_config import ActivationCollectionConfig
        return ActivationCollectionConfig
    if name == "SAELayerTrainingConfig":
        from geniesae.configs.training_config import SAELayerTrainingConfig
        return SAELayerTrainingConfig
    if name == "SAETrainingConfig":
        from geniesae.configs.training_config import SAETrainingConfig
        return SAETrainingConfig
    if name == "EvaluationConfig":
        from geniesae.configs.evaluation_config import EvaluationConfig
        return EvaluationConfig
    if name == "SAELightningModule":
        from geniesae.sae_lightning import SAELightningModule
        return SAELightningModule
    raise AttributeError(f"module 'geniesae' has no attribute {name!r}")

__all__ = [
    "ActivationCollectionConfig",
    "SAELayerTrainingConfig",
    "SAETrainingConfig",
    "EvaluationConfig",
    "TopKSAE",
    "SAELightningModule",
]
