import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Import data from csv
data = pd.read_csv(r'C:\Users\seanh\Downloads\healthcare-dataset-stroke-data.csv')

# Copy data for work
df = data.copy()
df.head()

# Drop unneeded column
df.drop(['id'], axis=1, inplace=True)
df.head()
df.describe()
df.info()


def isnull_table(data):
    """Count the number of null values in a dataframe per row as the total number and percentage values represented.

    Args:
    data(DataFrame): The DataFrame to count the null values in

    Returns:
    A table showing the total missing values and percentage those values make up."""

    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False) * 100
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

# Create isnull table for our DataFrame
isnull_table(df)

# Impute NaN values with mean BMI values
df=df.fillna(np.mean(df['bmi']))

# Separate data into numerical and categorical for EDA
df_num = df[['age', 'avg_glucose_level', 'bmi']]
df_cat = df[['stroke', 'smoking_status', 'Residence_type', 'work_type', 'ever_married', 'heart_disease', 'hypertension', 'gender']]

# Plot KDE plot of numerical data
fig, ax = plt.subplots(figsize= (10,6))
sns.kdeplot(data=df_num, fill=True)
plt.title('KDE Plot of Numerical Data', loc='center')
plt.xlabel('Distribution')
plt.ylabel('Densty')

# Plot boxplot of age vs. bmi
fig, ax = plt.subplots(figsize= (20,8))
sns.boxplot(x='age', y='bmi', data=df_num, whis=10)
plt.title('Boxplot: Age vs. BMI', loc='center')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()

# Plot boxplot of age vs avg_glucose_level
fig, ax = plt.subplots(figsize= (20,8))
sns.boxplot(x='age', y='avg_glucose_level', data=df_num, whis=10)
plt.title('Boxplot: Age vs. Average Glucose Level', loc='center')
plt.xlabel('Age')
plt.ylabel('Average Glucose Level')
plt.show()


def create_hists(data):
    """Plot histograms in a DataFrame by column.

    Args:
    data(DataFrame): The DataFrame to plot the histogram from.

    Returns:
    Multiple histograms, 1 for each column in the DataFrame.
    """
    fig = plt.figure(figsize=(20, 23))
    for indx, val in enumerate(data.columns):
        ax = plt.subplot(4, 3, indx + 1)
        ax.set_title(val, fontweight='bold')
        ax.grid(linestyle='--', axis='y')
        plt.hist(data[val], color='forestgreen', histtype='bar', align='mid')

# Plot histograms for our DataFrame
create_hists(df_cat)


def create_heatmap(data):
    """Plot heatmap of a Correlation Matrix of features in a DataFrame.

    Args:
    data(DataFrame): The DataFrame whose Correlation Matrix a heatmap is plotted of.

    Returns:
    A heatmap for a DataFrame's Correlation Matrix.
    """
    fig = plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    return sns.heatmap(data=correlation_matrix, annot=True, square=True, cmap='coolwarm')


# Plot heatmap of the Correlation Matrix between the features in our DataFrame
create_heatmap(df)

# Convert binary categorical features into 0's and 1's
df['gender']=df['gender'].apply(lambda x : 1 if x=='Male' else 0)
df["Residence_type"] = df["Residence_type"].apply(lambda x: 1 if x=="Urban" else 0)
df["ever_married"] = df["ever_married"].apply(lambda x: 1 if x=="Yes" else 0)

# Count instances of 'Unkown' in 'smoking_status'
data.isin(['Unknown']).sum(axis=0)

# Drop 'unkown' values in 'smoking_status' feature
df.drop(df.loc[df['smoking_status']=='Unknown'].index, inplace=True)

# Ensure rows with 'Unkown' are dropped
df['smoking_status'].unique()

# Employ One Hot encoding on features smoking_status and work_type
df_dummies = df[['smoking_status', 'work_type']]
df_dummies = pd.get_dummies(df_dummies)
df.drop(columns=['smoking_status', 'work_type'], inplace=True)
df=df.merge(df_dummies, left_index=True, right_index=True, how='left')

# Separate target variable from data
stroke_target = df['stroke']
stroke_data = df.drop(columns=['stroke'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(stroke_data, stroke_target, test_size=0.3, random_state=1)
# Split the data further into to create a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# Standardize our training, testing, and validation data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Instantiate our models
kn = KNeighborsClassifier()
log_reg = LogisticRegression()
dt = DecisionTreeClassifier()
svc = SVC()

# Instantiate a list of models
model_list = [kn, log_reg, dt, svc]


def score_model(model):
    """Iterate over a list of instantiated models to calculate the training and test scores of each model
        and append to a list model_score.

    Args:
    model(list): A list of instantiated models.

    Returns:
    A DataFrame containing the training and testing scores for each model.
    """
    model_score = []
    for model in model:
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        model_score.append([model, train_score, test_score])

    df_scores = pd.DataFrame.from_records(model_score, columns=['ML Model', 'Training Score', 'Testing Score'])
    return df_scores

# Create DataFrame of model training and testing scores
df_scores = score_model(model_list)
df_scores.head()


def model_eval(model):
    """Plot evaluation metrics for each model in a list of instantiated models.

    Args:
    model(list): A list of instantiated models to iterate over.

    Returns:
    Confusion Matrices and ROC Curves for each model in the list.
    """
    for model in model:
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        cm = confusion_matrix(y_test, prediction)
        plot_confusion_matrix(model, X_test, y_test)
        metrics.plot_roc_curve(model, X_test, y_test)

# Generate model evaluation metrics and visualizations
model_eval(model_list)


def hyperparam_tuning(model, params):
    """Use GridSearchCV to tune hyperparameters for a given model.

    Args:
    model(instantiated model): Instantiaed Machine Learning model.
    params(dict): Dictionary of parameters for GridSearchCV to try.

    Returns:
    Prints the best parameters, best score, and best estimator for the given model.
    """
    grid_model = GridSearchCV(model, param_grid=params, n_jobs=-1)
    grid_model.fit(X_train, y_train)
    best_params = grid_model.best_params_
    best_score = grid_model.best_score_
    best_model = grid_model.best_estimator_
    print('The best parameters for this model are: {}.'.format(best_params))
    print('The best score for this model is: {}.'.format(best_score))
    print('The best model is: {}.'.format(best_model))


def cross_val(model):
    """Iterate over a list of instaniated models to compute the Cross Validation Score.

    Args:
    model(list): List of instantiated machine learning algorithms.

    Returns:
    A DataFrame containing the cross validation scores, cross validation mean, and cross validation standard deviation for
    each model.
    """
    cross_val_scores = []
    for model in model:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        means = np.mean(cv_scores)
        stds = np.std(cv_scores)
        cross_val_scores.append([model, cv_scores, means, stds])
        cv_scores = pd.DataFrame.from_records(cross_val_scores, columns=['ML Model', 'CV Score', 'CV Mean', 'CV STD'])
    return cv_scores

# Create DataFrame of cross validation scores
cv_scores = cross_val(model_list)
cv_scores.head()

# Create parameter dictionary for KNeighborsClassifier
param_kn = {
    'leaf_size': np.arange(1, 50),
    'n_neighbors': np.arange(1, 30),
    'p': [1,2]
}

# Create parameter dictionary for DecisionTreeClassifier
param_dt = {"max_depth": [3, None],
              "max_features": np.arange(1, 9),
              "min_samples_leaf": np.arange(1, 9),
              "criterion": ["gini", "entropy"]}

# Create parameter dictionary for LogisticRegression
params_log_reg = {
    'C': [100, 10, 1.0, 0.1, 0.01],
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'penalty': ['l1', 'l2']
}

# Create parameter dictionary for SVC
param_svc = {
    'C': [0.1,1, 10, 100],
    'gamma': [1,0.1,0.01,0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Perform hyperparameter tuning for KNeighborsClassifier
hyperparam_tuning(kn, param_kn)

# Perform hyperparameter tuning for DecisionTreeClassifier
hyperparam_tuning(dt, param_dt)

# Perform hyperparameter tuning for LogisticRegression
hyperparam_tuning(log_reg, params_log_reg)

# Perform hyperparameter tuning for SVC
hyperparam_tuning(svc, param_svc)

# Instantiate tuned models
kn_tuned = KNeighborsClassifier(leaf_size=1, n_neighbors=6, p=1)
log_reg_tuned = LogisticRegression(C=100, penalty='l1', solver='liblinear')
dt_tuned = DecisionTreeClassifier(criterion='entropy', max_depth=3, max_features=7)
svc_tuned = SVC(C=0.1, gamma=1)

# Instantiate a list of tuned models
tuned_list = [kn_tuned, log_reg_tuned, dt_tuned, svc_tuned]

# Create DataFrame of tuned models training and testing scores
tuned_scores = score_model(tuned_list)

# Rename columns to signify tuned scores
tuned_scores.rename(columns={'ML Model': 'Tuned Model', 'Training Score': 'Tuned Training Score', 'Testing Score': 'Tuned Testing Score'}, inplace=True)
tuned_scores.head()

# Create DataFrame of tuned cross validation scores
tuned_cv_scores = cross_val(tuned_list)

# Rename columns to signify tuned scores
tuned_cv_scores.rename(columns={'ML Model': 'Tuned Model', 'CV Score': 'Tuned CV Score', 'CV Mean': 'Tuned CV Mean', 'CV STD': 'Tuned CV STD'}, inplace=True)
tuned_cv_scores.head()

# Merge untuned and tuned scores DataFrames
df_scores = df_scores.merge(tuned_scores, left_index=True, right_index=True, how='left')
df_scores.head()

# Merge untuned and tuned cross validation scores DataFrames
cv_scores = cv_scores.merge(tuned_cv_scores, left_index=True, right_index=True, how='left')
cv_scores.head()


def val_set_score(model):
    """Iterate over a list of tuned models to compute scores on the validation set.

    Args:
    model(list): A list of tuned and instantiaed models to iterate over.

    Returns:
    A DataFrame with scores for the validation set.
    """
    val_set_scores = []
    for model in model:
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        cv_scores = cross_val_score(model, X_val, y_val, cv=5)
        means = np.mean(cv_scores)
        stds = np.mean(cv_scores)
        val_set_scores.append([model, val_score, cv_scores, means, stds])
    df_val_set_scores = pd.DataFrame.from_records(val_set_scores,
                                                  columns=['ML Model', 'Val Score', ' Val CV Score', 'Val CV Mean',
                                                           'Val CV STD'])
    return df_val_set_scores

# Create a DataFrame of scores using the reserved validation set
df_val_set_scores = val_set_score(tuned_list)
df_val_set_scores.head()


def val_set_eval(model):
    """Plot evaluation metrics for the validation set for each model in a list of instantiated models.

    Args:
    model(list): A list of instantiated models to iterate over.

    Returns:
    Confusion Matrices and ROC Curves for each model in the list.
    """
    for model in model:
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        cm = confusion_matrix(y_val, prediction)
        plot_confusion_matrix(model, X_val, y_val)
        metrics.plot_roc_curve(model, X_val, y_val)

val_set_eval(tuned_list)