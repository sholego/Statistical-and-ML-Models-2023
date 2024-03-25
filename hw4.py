import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import cycle
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LassoLarsCV, LassoLarsIC, lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
import seaborn as sns

url = 'https://raw.githubusercontent.com/TaddyLab/BDS/master/examples/web-browsers.csv'
webbrowsers_df = pd.read_csv(url)
webbrowsers_df = webbrowsers_df[["id", "spend"]]
webbrowsers_df["spend_ln"] = np.log(webbrowsers_df["spend"])
del webbrowsers_df["spend"]

url = 'https://raw.githubusercontent.com/TaddyLab/BDS/master/examples/browser-domains.csv'
browserdomains_df = pd.read_csv(url)

total_visit_user_level = browserdomains_df.groupby(by=["id"]).sum().reset_index()

browserdomains_visits_spend = browserdomains_df.merge(total_visit_user_level[["visits", "id"]].rename(columns={"visits":"total_visits"}), on="id")

browserdomains_visits_spend["visitspercent"] = 100 * (browserdomains_visits_spend["visits"] / browserdomains_visits_spend["total_visits"])

browserdomains_visitpercent = browserdomains_visits_spend[["id", "site", "visitspercent"]]
browserdomains_visitpercent = browserdomains_visitpercent.pivot(index='id', columns='site', values='visitspercent').fillna(0).reset_index()

webbrowsers_df = webbrowsers_df.merge(browserdomains_visitpercent, on="id", how="inner")

y_labels = list(webbrowsers_df.columns)
y_labels.remove("id")
y_labels.remove("spend_ln")
X, y = webbrowsers_df[y_labels].values, webbrowsers_df["spend_ln"]
n_max_iter = 10**3*5

# LASSO Information Criteria
# calculation computing time
start_time = time.time()
lasso_lars_ic = make_pipeline(
    StandardScaler(with_mean=False), LassoLarsIC(criterion="aic", max_iter=n_max_iter)
).fit(X, y)

#AIC
min_aic_index =np.argmin(lasso_lars_ic[-1].criterion_)
min_aic = lasso_lars_ic[-1].criterion_.min()
aic_select_lambda = lasso_lars_ic[-1].alphas_[min_aic_index]

# BIC
lasso_lars_ic.set_params(lassolarsic__criterion="bic").fit(X, y)
alpha_bic = lasso_lars_ic[-1].alpha_
min_bic_index =np.argmin(lasso_lars_ic[-1].criterion_)
min_bic = lasso_lars_ic[-1].criterion_.min()
bic_select_lambda = lasso_lars_ic[-1].alphas_[min_bic_index]
fit_time_ic = time.time() - start_time


# LASSO k-Cross Varidattion (k=20)
# calculation computing time
start_time = time.time()
fit_time = time.time() - start_time
model = make_pipeline(StandardScaler(with_mean=False), LassoLarsCV(cv=20, max_iter=n_max_iter)
).fit(X, y)
fit_time = time.time() - start_time
lasso_cv = model[-1]

# Linear Regression with cross_val_score
linear_model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
linear_MSE_scores = cross_val_score(linear_model, X, y, scoring='neg_mean_squared_error', cv=20)
linear_MSE_scores *= -1

# Get the best alpha and MSE from LassoLarsCV
best_alpha_idx = np.argmin(lasso_cv.mse_path_.mean(axis=-1))
lasso_MSE_scores = lasso_cv.mse_path_[best_alpha_idx,]

# Create new Dataframe
data_Q1 = pd.DataFrame({'Linear Regression': linear_MSE_scores, 'Lasso': lasso_MSE_scores})

plt.figure(figsize=(12, 6))
sns.boxplot(data = data_Q1)
plt.xlabel('Model')
plt.ylabel('MSE')
plt.savefig('hw4_Q1.png')
plt.show()
#-------------------------Q2-------------------------------------#
# Linear Regression with cross_val_score using R^2 scoring
linear_r2_scores = cross_val_score(linear_model, X, y, scoring='r2', cv=20)

# Get the R^2 scores from LassoLarsCV
lasso_r2_scores = 1 - lasso_cv.mse_path_[best_alpha_idx,] / np.var(y)

# Create new DataFrame for R^2 scores
data_Q2 = pd.DataFrame({'Linear Regression': linear_r2_scores, 'Lasso': lasso_r2_scores})

# Plotting
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_Q2)
plt.xlabel('Model')
plt.ylabel('R^2 Score')
plt.savefig('hw4_Q2.png')
plt.show()
#-------------------------Q3-------------------------------------#
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Convert webbrowsers_df spendln to spend
webbrowsers_df['spend'] = np.exp(webbrowsers_df['spend_ln'])

# Split y and X data and convert to binary variables
y_Q3 = (webbrowsers_df['spend'] > 1000).values # more than $1000
X_Q3 = webbrowsers_df[[i for i in range(1, 1001)]].values

def evaluate_logistic_regression(X, y, n_calc, **log_reg_params):
    accuracies_cheat = []
    accuracies_valid = []
    for i in range(n_calc):

        # ===========Repeating n_calc times==============
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression()
        model.fit(X_train, y_train) # Learning weights for logistic regression model
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        # Making predictions and evaluating the model
        accuracies_cheat.append({"accuracy":accuracy_score(y_true=y_train, y_pred=y_pred_train), "label": "cheat"})
        accuracies_valid.append({"accuracy":accuracy_score(y_true=y_test, y_pred=y_pred_test), "label": "valid"})

    return accuracies_cheat, accuracies_valid

score_cheat, score_valid = evaluate_logistic_regression(X_Q3, y_Q3, 100, n_splits=100, 
                                                          max_iter=10**5)

df_Q3_cheat = pd.DataFrame(score_cheat)
df_Q3_valid = pd.DataFrame(score_valid)
df_Q3 = pd.concat([df_Q3_cheat, df_Q3_valid], axis=0)

plt.figure(figsize=(12, 6))
order = ["cheat", "valid"]
sns.boxplot(x="label", y="accuracy", data=df_Q3)
plt.xlabel("label")
plt.ylabel("accuracy")
plt.title("accuracy_Boxplot")
plt.savefig("hw4_Q3_boxplot.png")
plt.show()
plt.close()

# Create histogram
plt.figure(figsize=(12, 6))
sns.histplot(data=df_Q3_cheat, x='accuracy', bins=20, kde=True, stat='density', common_norm=False, color='red', label='cheat')
sns.histplot(data=df_Q3_valid, x='accuracy', bins=20, kde=True, stat='density', common_norm=False, color='green', label='valid')
plt.title('Accuracy Distribution - Cheat vs. Valid')
plt.xlabel('Accuracy')
plt.ylabel('Density')
plt.legend(title='Label')
plt.savefig('hw4_Q3_histogram.png')
plt.show()
#-------------------------Bonus1-------------------------------------#
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_MSE_scores = cross_val_score(rf_model, X, y, cv=20, scoring='neg_mean_squared_error')
rf_MSE_scores *= -1

# Summarize scores into a DataFrame
df_bonus1 = pd.DataFrame({
     'Model': ['Linear Regression'] * len(linear_MSE_scores) + ['Random Forest'] * len(rf_MSE_scores),
     'MSE': np.concatenate([linear_MSE_scores, rf_MSE_scores])
 })

sns.set(style="whitegrid")
order = ["Linear Regression", "Random Forest"]
plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='MSE', data=df_bonus1, palette={'Linear Regression': 'lightblue', 'Random Forest': 'lightcoral'}, width=0.5, order=order)
plt.xlabel('Model')
plt.ylabel('MSE')
plt.savefig('hw4_bonus1.png')
plt.show()
#-------------------------Bonus2（Failed）-------------------------------------#
num_outliers = 100
np.random.seed(42) 
outliers_X = np.random.normal(loc=100, scale=30, size=(num_outliers, X.shape[1]))
outliers_y = np.random.normal(loc=150, scale=50, size=num_outliers) 

# Add outliers to original data
X_with_outliers = np.vstack([X, outliers_X])
y_with_outliers = np.concatenate([y, outliers_y])

# Get the worst alpha
worst_alpha_idx = np.argmax(lasso_cv.mse_path_.mean(axis=-1))
lasso_MSE_scores_worst = lasso_cv.mse_path_[worst_alpha_idx,]

# Linear Regression with cross_val_score
linear_model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
linear_MSE_scores = cross_val_score(linear_model, X, y, scoring='neg_mean_squared_error', cv=20)
linear_MSE_scores *= -1

# Summarize scores into a DataFrame
df_bonus2 = pd.DataFrame({
    'Model': ['Linear Regression_withoutOL'] * len(linear_MSE_scores) + ['Lasso_withOL'] * len(lasso_MSE_scores_worst),
    'MSE': np.concatenate([linear_MSE_scores, lasso_MSE_scores_worst])
})


sns.set(style="whitegrid")
order = ["Linear Regression_withoutOL", "Lasso_withOL"]
plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='MSE', data=df_bonus2, palette={'Linear Regression_withoutOL': 'lightblue', 'Lasso_withOL': 'lightcoral'}, width=0.5, order=order)
plt.xlabel('Model')
plt.ylabel('MSE')
plt.savefig('hw4_bonus2.png')
plt.show()
