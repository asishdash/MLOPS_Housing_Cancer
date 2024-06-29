import mlflow
import numpy
import pandas as pd
from sklearn.datasets import fetch_openml

housing = fetch_openml(name="house_prices", as_frame=True)

temp = housing.data[['OverallQual',
        'GrLivArea',
        'GarageCars',
        'GarageArea',
        'TotalBsmtSF',
        '1stFlrSF',
        'FullBath',
        'TotRmsAbvGrd',
        'YearBuilt']]

data = numpy.c_[temp, housing.target]
columns = ['OverallQual',
        'GrLivArea',
        'GarageCars',
        'GarageArea',
        'TotalBsmtSF',
        '1stFlrSF',
        'FullBath',
        'TotRmsAbvGrd',
        'YearBuilt',
        'target']
# columns = numpy.append(temp.feature_names, ["target"])
print("hello1")
housing_data = pd.DataFrame(data, columns=columns)
print("hello")
# data = pd.DataFrame(housing)
# print(type(housing))
print(housing_data.head())
with open('reference_data.csv', 'a') as f:
        housing_data.to_csv(f, index=False)

# all_runs = mlflow.search_runs(search_all_experiments=True)
# print(all_runs[['run_id','metrics.r2']])
# print(all_runs.info())
# best_model = mlflow.search_runs(filter_string="metrics.Precision>0.95")
# print(best_model.info())
# print(best_model)
# print(best_model[['run_id','metrics.Precision']])

# model = mlflow.pyfunc.load_model("runs:/bdca5d0a1d184367a792be0631749684/Mean Run")
# model.serve(port=5151)

'''
py -m venv MLFLowEnv
cd .\MLFlowEnv\
.\MLFlowEnv\Scripts\Activate.ps1

pip install -r requirements.txt

Uncomment: 63, 73, 95, 96
Comment: 66, 74, 85, 97, 98
py -m program.py
====================================================
Uncomment: 66, 74, 85, 97, 98
Comment: 63, 73, 95, 96
py -m program

user only first 4 lines from postcode.py and run the following command
py -m postcode

copy the run_id for model with highest R2 and artifact_uri
paste this in line 13 of app.py

py -m streamlit run app.py
'''
