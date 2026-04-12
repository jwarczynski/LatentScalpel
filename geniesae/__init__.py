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
    if name == "TemporalClassifier":
        from geniesae.temporal_classifier import TemporalClassifier
        return TemporalClassifier
    if name == "TemporalProfile":
        from geniesae.temporal_classifier import TemporalProfile
        return TemporalProfile
    if name == "FeatureInterventionPatcher":
        from geniesae.feature_intervention import FeatureInterventionPatcher
        return FeatureInterventionPatcher
    if name == "ScheduleModifier":
        from geniesae.schedule_modifier import ScheduleModifier
        return ScheduleModifier
    if name == "InterventionConfig":
        from geniesae.configs.intervention_config import InterventionConfig
        return InterventionConfig
    if name == "ScheduleExperimentConfig":
        from geniesae.configs.schedule_experiment_config import ScheduleExperimentConfig
        return ScheduleExperimentConfig
    raise AttributeError(f"module 'geniesae' has no attribute {name!r}")

__all__ = [
    "ActivationCollectionConfig",
    "SAELayerTrainingConfig",
    "SAETrainingConfig",
    "EvaluationConfig",
    "TopKSAE",
    "SAELightningModule",
    "TemporalClassifier",
    "TemporalProfile",
    "FeatureInterventionPatcher",
    "ScheduleModifier",
    "InterventionConfig",
    "ScheduleExperimentConfig",
]
