import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import requests
from io import BytesIO
import zipfile

def get_excel():
    url = "https://archive.ics.uci.edu/static/public/437/residential+building+data+set.zip"
    response = requests.get(url)
    zip_data = BytesIO(response.content)
    with zipfile.ZipFile(zip_data, 'r') as z:
        xlsx_files = [name for name in z.namelist() if name.endswith('.xlsx')]
        d = {file_name: pd.read_excel(z.open(file_name), skiprows=1) for file_name in xlsx_files}
    return d

#-------------Function Definition--------------#
def calc_r2_r2adj(data, ex_variable_names, N_list):
    # Specifying the Objective Variable
    target_variable = "V-10"
    # Initializing a list to store the result
    results_r2 = []
    # Extracting the objective variable
    Y = data[target_variable]
    
    
    for N in N_list:
        for i in range(100):
            # Creation of a sample of randomly selected explanatory variables
            selected_vars = shuffle(ex_variable_names, n_samples=N, random_state=None)
            # Extract data for N randomly selected explanatory variables and add intercept
            X = data[selected_vars]
            X = sm.add_constant(X)

            # Build a regression model
            model = sm.OLS(Y, X)
            results = model.fit()
            
            # Store the results of the regression model in result_r2
            # Collect the results of the regression model for the number of different explanatory variables (N) 
            # and include the modified R-squared and R-squared values in the results
            
            # Represents the normal R-squared
            results_r2.append({"n_variables": N, "r2": results.rsquared, "adj": False})
            # Represents the adjusted R-squared
            results_r2.append({"n_variables": N, "r2": results.rsquared_adj, "adj": True}
            )
    # Convert the results to a DataFrame and return
    return pd.DataFrame(results_r2)

# Function to read an Excel file
air_quality_data = get_excel()['Residential-Building-Data-Set.xlsx']
# Name of explanatory variable
ex_variable_names = air_quality_data.columns[4:-2].values

# Create N_list [10,20,30,40,50,60,70,80,90,100]
n_values = list(range(10, 101, 10))

# Substitute the prepared data and N_list into the function (the part where the calculator computes the data)
df = calc_r2_r2adj(data=air_quality_data, ex_variable_names=ex_variable_names, N_list=n_values)

sns.set_theme(style="ticks", palette="pastel")
# data_for_boxplot = pd.DataFrame({'n_variables': positions, 'r2': r2_lists})
sns.boxplot(x="n_variables", y="r2", hue="adj", data=df, palette=["m", "g"])
sns.despine(offset=10, trim=True)

plt.show()
