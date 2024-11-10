# Project: Bank Customer Churn Prediction: API
#### This is an API for the supervised learning models training and inference for the Bank customer churn dataset.
#### Course: Big Data, Machine Learning, with Applications to Economics and Finance
#### Student: Sofya Cheburkova, 2nd year Masters, e-mail: sacheburkova@edu.hse.ru

## Description
### Attributes
This API is designed to provide predictions on the likelihood of a customer exiting (churning) from a bank. 
The model is trained on a dataset of 10 000 banking customers (taken from: www.kaggle.com - 'Bank_Churn_Modelling.csv') with various customer attributes, such as:
- Credit Score: A numerical value representing the customer's credit score
- Geography: The country where the customer resides (France, Spain or Germany)
- Gender: The customer's gender (Male or Female)
- Age: The customer's age
- Tenure: The number of years the customer has been with the bank
- Balance: The customer's account balance
- NumOfProducts: The number of bank products the customer uses (e.g., savings account, credit card)
- HasCrCard: Whether the customer has a credit card (1 = yes, 0 = no)
- IsActiveMember: Whether the customer is an active member (1 = yes, 0 = no)
- EstimatedSalary: The estimated salary of the customer

For example, a customer's credit score reflects their financial behavior, where lower scores may indicate dissatisfaction
and a higher likelihood of leaving. Longer tenure often correlates with loyalty, typically lowering churn rates, while
balance trends can also signal risk; higher balances imply satisfaction, whereas declining balances suggest potential churn.

### Models
I used Logistic Regression and Random Forest classifiers to assess the probability of churn.
- LogReg was chosen for its interpretability and effectiveness in binary classification tasks like predicting churn.
It also makes visible the impact of each input feature on the churn probability, making it easier to interpret results.
- RF was selected for its robustness and ability to handle complex, nonlinear relationships within big datasets.
It allows to capture interactions between features that may not be obvious, leading to potentially more accurate predictions.

It is important to note that traditional models, such as regression, can struggle with high-dimensional data due to overfitting, multicollinearity and complexity in selecting which features are most important.
The technique like Random Forest can handle large datasets, such as the underlying one with 10 000 customers, are robust to missing data and can handle noise in the dataset.
It can efficiently process a high number of features to identify subtle trends as well.

## Requirements
Requirements are specified in the requirements.txt file. Ensure all packages are installed before executing the code.

## Installation & Setup
- Clone the Repository: Start by cloning the repository to your local machine.
- Virtual Environment: Open the project in any IDE, ex.: PyCharm. Create a virtual environment within PyCharm or using the command line to manage dependencies separately from your system environment.
- Install Dependencies: Install the necessary packages from requirements.
- Starting the API: Run the main application file app.py to start the API server. 

## Quick Start
To quickly get the API running, ensure that your virtual environment is active and run the app.py file. 
By default, the server will start locally on port 8000. You can access the API and Swagger documentation at http://127.0.0.1:8000/docs.

## Testing
Example Prediction: To test a prediction, send a POST request to the /predict endpoint with the model and input features as JSON. 
You can also use test_api.ipynb to interact with the API directly in Jupyter Notebook.

## API Endpoints:
- /list_models: returns a list of available models with their associated hyperparameters.
- /train_model: trains the specified model using hyperparameter tuning via GridSearchCV.
- /predict: provides churn probability predictions for bank customers based on the input features.
- /model_delete/{model_class}: deletes the specified model from memory.

## Error Handling & Limitations
In this code, error handling ensures robust interactions with the API and provides clear feedback when issues arise. 
When making predictions, the API first verifies if the specified model (logistic regression 'lr' or random forest 'rf') has been trained. 
If a model is untrained, the API returns an error message prompting the user to train the model first, with a 500 status code. 
Similarly, if an unsupported model type is requested, an error message notifies users that only 'lr' or 'rf' models are available. 
Finally, any other unexpected errors are caught and logged and an error response is sent with the exception details to aid in troubleshooting.
