from pydantic import BaseModel

class AvailableModel(BaseModel):
    name_model: str
    description: str
    hyperparams: list[dict]

class HyperInput(BaseModel):
    name: str
    min_value: int
    max_value: int
    steps: int
    log_scale: bool
    is_int: bool

class TrainInput(BaseModel):
    data: list[list]
    name_model: str
    hyp_range: list[HyperInput]

class TrainOutput(BaseModel):
    metric: float
    metric_name: str
    best_params: dict[str, float]
    #time: float

class InferenceInput(BaseModel):
    name_model: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class InferenceOutput(BaseModel):
    Exited_rate: float


