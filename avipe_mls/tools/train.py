import argparse

from .. import trainer
from .. import config
from .. import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Avipe mls trainer tool",
        description="Tool to train models"
    )
    parser.add_argument("-c", '--config', help="Path to config.yaml file", required=True)
    parser.add_argument("-r", "--root-path", help="Root path to experiment weights directory", default=constants.DEFAULT_ROOT_PATH)
    args = parser.parse_args()
    
    full_config = config.load(args.config)
    trainer.train_model(full_config, args.root_path)