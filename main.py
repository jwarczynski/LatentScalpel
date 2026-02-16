import argparse

from geniesae.config import ExperimentConfig
from geniesae.pipeline import PipelineRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run the GENIE SAE training and evaluation pipeline."
    )
    parser.add_argument("config", help="Path to experiment config YAML")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    config.validate()
    runner = PipelineRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
