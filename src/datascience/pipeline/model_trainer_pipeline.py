from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience import logger

STAGE_NAME="Model Trainer Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        config=ConfigurationManager()
        data_validation_config=config.get_model_trainer_config()
        data_validation=ModelTrainer(config=data_validation_config)
        data_validation.train()


if __name__=="__main__":

    try:
        logger.info(f"stage {STAGE_NAME} started.....")
        obj = ModelTrainingPipeline()
        obj.initiate_model_training()
        logger.info(f"stage {STAGE_NAME} completed!")
    except Exception as e:
        raise e

