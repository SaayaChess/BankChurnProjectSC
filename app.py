import logging
import sys

import uvicorn
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from models import create_train, hyprer2dict, train_models, create_inferdf, inference
from schemas import AvailableModel, TrainOutput, TrainInput, InferenceOutput, InferenceInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    "%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

description = """
API for the training and supervised models inference for the bank customer churn dataset.
"""

model_lr = None
model_rf = None

app = FastAPI(title="BankChurn API",
              description=description,
              summary="API for Bank Customer Churn",
              version="0.0.1",
              terms_of_service="http://example.com/terms/",
              contact={
                  "name": "Sofya Cheburkova (ML Course)",
                  "email": "sacheburkova@edu.hse.ru",
              },
              license_info={
                  "name": "Apache 2.0",
                  "identifier": "MIT",
              })


@app.get('/list_models', response_model=list[AvailableModel])
async def list_model():
    model_list = []
    model1 = AvailableModel(name_model='log_reg',
                            description="Logistic regression ...",
                            hyperparams=[{
                                'name': 'C',
                                'description': "L2 reg coeff.",
                                'range': "0.001 - 1000"
                            }])
    model_list.append(model1)
    model2 = AvailableModel(name_model='rf',
                            description="Random Forest ...",
                            hyperparams=[{
                                'name': 'n_estimators',
                                'description': "No. of base estimators.",
                                'range': "100 - 500"
                            },
                                {'name': 'max_depth',
                                 'description': "Max. depth of base estimator.",
                                 'range': "5 - 50"}])
    model_list.append(model2)
    return model_list


@app.post("/train_model", response_model=TrainOutput)
async def train_model(train_input: TrainInput):
    global model_lr
    global model_rf
    try:
        df = create_train(train_input.data)
        params_objs = train_input.hyp_range
        params_dict = hyprer2dict(params_objs)
        model_class = train_input.name_model
        # t0 = time.time()
        result, best_params, best_estimator = train_models(df, model_class, params_dict)
        # dt = time.time() - t0
        if model_class == 'lr':
            model_lr = best_estimator
        else:
            model_rf = best_estimator
        return TrainOutput(metric=result, metric_name='f1_weighted', best_params=best_params)
    except Exception as e:
        logger.error(e)
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/predict", response_model=InferenceOutput)
async def inference_model(infer_input: InferenceInput):
    global model_lr
    global model_rf
    try:
        df = create_inferdf(infer_input)
        model_class = infer_input.name_model
        if model_class == 'lr':
            if model_lr is None:
                return JSONResponse(content={"error": "Firstly, the lr must be trained!"},
                                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return inference(model_lr, df)
        elif model_class == 'rf':
            if model_rf is None:
                return JSONResponse(content={"error": "Firstly, rf must be trained!"},
                                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return inference(model_rf, df)
        else:
            return JSONResponse(content={"error": "lr or rf models are supported only!"},
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        logger.error(e)
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.delete("/model_delete/{model_class}")
async def delete_person(model_class: str):
    global model_lr
    global model_rf
    if model_class == 'lr':
        if model_lr is not None:
            model_lr = None
        return JSONResponse(content={"message": "lr was deleted successfully."},
                            status_code=status.HTTP_200_OK)
    elif model_class == 'rf':
        if model_rf is not None:
            model_rf = None
        return JSONResponse(content={"message": "rf was deleted successfully."},
                            status_code=status.HTTP_200_OK)
    else:
        return JSONResponse(content={"error": "lr or rf models are supported only!"},
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', workers=1)
