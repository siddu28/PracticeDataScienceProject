import os
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_error,r2_score
import mlflow
import numpy as np
import joblib
from pathlib import Path
from src.datascience.config.configuration import ModelEvaluationConfig
from src.datascience.utils.common import save_json

# import os
# os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/siddukasam28/PracticeDataScienceProject.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "siddukasam28"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "63c4cbbeb538d23123b3424e131fc50146474157"

class ModelEvaluation:
    def __init__(self,config=ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse,mae,r2
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column],axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)
            (rmse,mae,r2) = self.eval_metrics(test_y,predicted_qualities)

            scores = {"rmse":rmse,"mae":mae,"r2":r2}
            save_json(path = Path(self.config.metric_file_name),data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse",rmse)
            mlflow.log_metric("r2",r2)

            os.makedirs("model",exist_ok=True)
            joblib.dump(model,"model/model.pkl")

            mlflow.log_artifacts("model",artifact_path="model")