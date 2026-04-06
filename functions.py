#IMPORT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,confusion_matrix
from sklearn.model_selection import train_test_split , cross_validate , validation_curve , GridSearchCV , ValidationCurveDisplay
from sklearn.preprocessing import StandardScaler , MinMaxScaler, LabelEncoder , OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier , NearestCentroid
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
# Models
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
# eda
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
#general
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#PRINT
def PRINT(text):
  '''a nicer print , text - text to print
      example:
      --------
      text
      -------- '''
  length = len(text)
  if length < 140:
    length = 140
  print("-"*length)
  print(text)
  print("-"*length)
#SAMPLING/BALANCING
def undersampling(X,y):
  '''function for undersampling, X - features, y - target
  '''
  rus = RandomUnderSampler(sampling_strategy="majority",random_state = 42)
  new_X, new_y = rus.fit_resample(X, y)
  return new_X, new_y
def oversampling(X,y):
  '''function for overampling , X - features, y - target
  '''
  smote = SMOTE(sampling_strategy="minority",random_state = 42)
  new_X, new_y = smote.fit_resample(X,y)
  return new_X, new_y
#ENCODING
def label_encoding(df ,column,copy = False ,encoder = None):
  '''function for lable encoding, df - dataframe, column - column to encode, copy - if you want return a copy
  '''
  if copy:
    df1 = df.copy()
    encoder = LabelEncoder()
    df1[column] = encoder.fit_transform(df1[column])
    return df1
  if encoder is None:
      encoder = LabelEncoder()
      df[column] = encoder.fit_transform(df[column])
  else:   
     df[column] = encoder.transform(df[column])
  return df , encoder
def one_hot_encoding(df,list,copy = False , encoder = None):
  '''function for one hot encoding, df - dataframe, list - list of columns to encode, copy - if you want to return a copy
  '''
  if copy:
    df1 = df.copy()
    if encoder is None:
      encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df1[list])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(list), index=df1.index)
    df1 = pd.concat([df1, encoded_df], axis=1)
    df1 = df1.drop(list, axis=1)
    return df1
  if encoder is None:
      encoder = OneHotEncoder(sparse_output=False)
      encoded = encoder.fit_transform(df[list])
  else:
      encoded = encoder.transform(df[list])
  encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(list), index=df.index)
  df = pd.concat([df, encoded_df], axis=1)
  df = df.drop(list, axis=1)
  return df , encoder
#OUTLINERS
def outliners(df,column):
  '''function for outliners, df - dataframe, column - column to check,copy - if you want to return a copy
  '''
  q1 = df.describe().iloc[4 ,:][column]
  q3 = df.describe().iloc[6 , :][column]
  iqr = q3 - q1
  lower = q1 - 1.5 * iqr
  upper = q3 + 1.5 * iqr
  count = df[(df[column] < lower) | (df[column] > upper)][column].count()
  return(f"number of outlinears in {column} is {count}" , count)
def removeoutliners(df,column,copy = False):
  '''function to remove outliners, df - dataframe, column - column to check,copy - if you want to return a copy
  '''
  if copy:
    df1 = df.copy()
    q1 = df1.describe().iloc[4 ,:][column]
    q3 = df1.describe().iloc[6 , :][column]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    index = df1[(df1[column] < lower) | (df1[column] > upper)][column].index
    df1.drop(index , inplace = True,axis = 0 )
    return df1
  q1 = df.describe().iloc[4 ,:][column]
  q3 = df.describe().iloc[6 , :][column]
  iqr = q3 - q1
  lower = q1 - 1.5 * iqr
  upper = q3 + 1.5 * iqr
  index = df[(df[column] < lower) | (df[column] > upper)][column].index
  df.drop(index , inplace = True,axis = 0 )
  return df
def squashoutliners(df,column, copy = False):
  ''' function to squash outliners, df - dataframe, column - column to check,copy - if you want to return a copy
  '''
  if copy:
    df1 = df.copy()
    q1 = df1.describe().iloc[4 ,:][column]
    q3 = df1.describe().iloc[6 , :][column]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df1.loc[df[column] < lower , column] = lower
    df1.loc[df[column] > upper , column] = upper
    return df1
  q1 = df.describe().iloc[4 ,:][column]
  q3 = df.describe().iloc[6 , :][column]
  iqr = q3 - q1
  lower = q1 - 1.5 * iqr
  upper = q3 + 1.5 * iqr
  df.loc[df[column] < lower , column] = lower
  df.loc[df[column] > upper , column] = upper
  return df
#SCALING
def standardscaler(df,list,copy =False ,scaler = None):
  '''function to standard scaler, df - dataframe, list - list of columns to scale,copy - if you want to return a copy
  '''
  if copy:
    df1 = df.copy()
    scaler = StandardScaler()
    df1[list] = scaler.fit_transform(df1[list])
    return df1
  if scaler is None:
    scaler = StandardScaler()
    df[list] = scaler.fit_transform(df[list])
  else:
    df[list] = scaler.transform(df[list])
  return df , scaler
def minmaxscalar(df,list,copy =False,scaler = None):
  '''function to minmax scaler, df - dataframe, list - list of columns to scale,copy - if you want to return a copy
  '''
  if copy:
    df1 = df.copy()
    scaler = MinMaxScaler()
    df1[list] = scaler.fit_transform(df1[list])
    return df1
  if scaler is None:
    scaler = MinMaxScaler()
    df[list] = scaler.fit_transform(df[list])
  else:
    df[list] = scaler.transform(df[list])
  return df , scaler
#NULLS
import ipywidgets as widgets
from IPython.display import display, clear_output
import seaborn as sns
import pandas as pd
import numpy as np
def compare_imputation_methods(df, column_name):
    title_fontsize = 20
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    sns.histplot(df[column_name], kde=True, ax=axes[0])
    axes[0].set_title('ירוקמ', fontsize=title_fontsize)
    mean_imputed = df[column_name].fillna(df[column_name].mean())
    sns.histplot(mean_imputed, kde=True, ax=axes[1])
    axes[1].set_title('עצוממ', fontsize=title_fontsize)
    median_imputed = df[column_name].fillna(df[column_name].median())
    sns.histplot(median_imputed, kde=True, ax=axes[2])
    axes[2].set_title('ןויצח', fontsize=title_fontsize)
    mask = df[column_name].isnull()
    uniform_imputed = df[column_name].copy()
    uniform_imputed[mask] = np.random.uniform(df[column_name].min(), df[column_name].max(), size=mask.sum())
    sns.histplot(uniform_imputed, kde=True, ax=axes[3])
    axes[3].set_title('הדיחא תוגלפתה', fontsize=title_fontsize)
    normal_imputed = df[column_name].copy()
    normal_imputed[mask] = np.random.normal(df[column_name].mean(), df[column_name].std(), size=mask.sum())
    sns.histplot(normal_imputed, kde=True, ax=axes[4])
    axes[4].set_title('תילמרונ תוגלפתה', fontsize=title_fontsize)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
def on_column_selected(change, df):
    if change.new:
        clear_output(wait=True)
        display(column_selector)
        selected_column = change.new
        compare_imputation_methods(df, selected_column)
def create_column_selector(df):
    numeric_cols_with_na = [col for col in df.columns if df[col].isna().any() and pd.api.types.is_numeric_dtype(df[col])]
    global column_selector
    column_selector = widgets.Dropdown(
        options=numeric_cols_with_na,
        description='Select Column:',
        disabled=False,
    )
    column_selector.observe(lambda change: on_column_selected(change, df), names='value')
    display(column_selector)
def fill_uniform(df,column,copy = False):
  '''function to fill nulls with uniform distribution, df - dataframe, column - column to fill,copy - if you want to return a copy
  '''
  if copy:
    df1 = df.copy()
    possible_values = df1[column].dropna().unique()
    probabilities = [1/len(possible_values)] * len(possible_values)
    missing_indices = df1[column][df1[column].isnull()].index
    df1.loc[missing_indices, column] = pd.Series(np.random.choice(possible_values,
    size=len(missing_indices), p=probabilities), index=missing_indices)
    return df1
  possible_values = df[column].dropna().unique()
  probabilities = [1/len(possible_values)] * len(possible_values)
  missing_indices = df[column][df[column].isnull()].index
  df.loc[missing_indices, column] = pd.Series(np.random.choice(possible_values,
  size=len(missing_indices), p=probabilities), index=missing_indices)
  return df
def null(df):
  '''function to check nulls, df - dataframe
  '''
  return df.isnull().sum()
#MODEL
def train_test(x,y,test_size = 0.2,random_state=42):
  ''' function to split data into train and test ,x - features, y - target, test_size - test size, random_state - random state
  '''
  x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = test_size , random_state = random_state)
  return x_train,x_test,y_train,y_test
def evaluate(X_train,y_train,model_name,model,metrics,k =4,n_jobs=-1,verbose = 0):
  ''' function for evaluating model,model_name - name of the model,model - model, metrics - metrics to use,k - number of validation set,n_jobs - number of cpu used,verbose - verbose
  '''
  if verbose != 0:
    n_jobs = 1
  kf = KFold(n_splits=k, random_state=42, shuffle=True)
  scores = pd.DataFrame(cross_validate(model, X_train, y_train, cv=kf, scoring=metrics,return_train_score = True,n_jobs= n_jobs,verbose = verbose))
  for i in metrics:
    PRINT(f"{model_name} {i} score is {scores['test_' + i].mean()} on the test set and {scores['train_' + i].mean()} on the train set")
def knn(X_train,X_test,y_train,y_test,metric,k=4, params = {"n_neighbors": range(1,100)},n_jobs =-1,verbose = 0):
  ''' function for best knn model using gridsearch , X_train - train features, X_test - test features, y_train - train target, y_test - test target,metric - metrics to use,k  - number of validation set, params - parameters to use,n_jobs - number of cpu used,verbose - verbose
  '''
  if verbose != 0:
    n_jobs = 1
  if k !=0:
    kf = KFold(n_splits=k, random_state=42, shuffle=True)
  else:
    kf = None
  model = KNeighborsClassifier()
  grid_search = GridSearchCV(model, param_grid=params, cv=kf, scoring=metric,return_train_score= True,n_jobs=n_jobs,verbose = verbose)
  grid_search = grid_search.fit(X_train,y_train)
  best_k = grid_search.best_params_["n_neighbors"]
  model = KNeighborsClassifier(n_neighbors=best_k)
  model.fit(X_train,y_train)
  grid = pd.DataFrame(grid_search.cv_results_)
  best_train_score = grid.loc[grid_search.best_index_, "mean_train_score"]
  #PRINT(f"k = {best_k} gives the best knn model with a {metric} of {grid_search.best_score_} on the test(validation) set and a {metric} of {best_train_score} on the training set")
  return model ,grid
def bagging(base_name,base, X_train,X_test,y_train,y_test,metric,k=4,params = {"n_estimators":range(1,101)},n_jobs =-1,verbose = 0):
  ''' function for best bagging model using gridsearch,base_name - name of the base model,base - base model, X_train - train features, X_test - test features, y_train - train target, y_test - test target,metric - metrics to use,k  - number of validation set, params - parameters to use,n_jobs - number of cpu used,verbose - verbose
  '''
  if verbose != 0:
    n_jobs = 1
  kf = KFold(n_splits=k, random_state=42, shuffle=True)
  model = BaggingClassifier(estimator=base,random_state=42)
  grid_search = GridSearchCV(model, params,cv =kf, scoring=metric,return_train_score= True,n_jobs = n_jobs,verbose = verbose)
  grid_search = grid_search.fit(X_train,y_train)
  best_n_estimators = grid_search.best_params_["n_estimators"]
  best_model = BaggingClassifier(estimator=base,n_estimators=best_n_estimators)
  best_model.fit(X_train,y_train)
  grid = pd.DataFrame(grid_search.cv_results_)
  best_train_score = grid.loc[grid_search.best_index_, "mean_train_score"]
  PRINT(f"n_estimators = {best_n_estimators} gives the best {base_name} bagging  model with a {metric} of {grid_search.best_score_} on the test(validation) set and a {metric} of {best_train_score} on the training set")
  return best_model ,grid
def ncc(X_train,X_test,y_train,y_test,metric):
  ''' function for ncc model, X_train - train features, X_test - test features, y_train - train target, y_test - test target, metric - metric to use
  '''
  model = NearestCentroid()
  model.fit(X_train,y_train)
  evaluate("ncc",model,metric)
  return model
def tree(X_train,X_test,y_train,y_test,metric,k= 4,params = {"max_depth": [10,20,30,50,90,100,None],"min_samples_leaf":range(1,21)},n_jobs =-1,verbose = 0):
  ''' function for best tree model using gridsearch ,X_train - train features, X_test - test features, y_train - train target, y_test - test target, metric - metric to use,k  - number of validation set, params - parameters to use,n_jobs -number of cpu used,verbose - verbose
  '''
  if verbose != 0:
    n_jobs = 1
  kf = KFold(n_splits=k, random_state=42, shuffle=True)
  model = DecisionTreeClassifier(random_state = 42)
  grid_search = GridSearchCV(model, param_grid=params, cv=kf, scoring=metric,return_train_score= True,n_jobs = n_jobs,verbose = verbose)
  grid_search = grid_search.fit(X_train,y_train)
  best_max_depth = grid_search.best_params_["max_depth"]
  best_min_samples_leaf = grid_search.best_params_["min_samples_leaf"]
  model = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf,random_state = 42)
  model.fit(X_train,y_train)
  grid = pd.DataFrame(grid_search.cv_results_)
  best_train_score = grid.loc[grid_search.best_index_, "mean_train_score"]
  PRINT(f"max_depth = {best_max_depth}, min_samples_leaf = {best_min_samples_leaf} gives the best tree model with a {metric} of {grid_search.best_score_} on the test(validation) set and a {metric} of {best_train_score} on the training set")
  return model ,grid
def random_forest(X_train,X_test,y_train,y_test,metric,k=4, params1 = {"max_depth": [10,20,30,50,90,100,None],"min_samples_leaf":range(1,21)}, params2 = {"n_estimators":range(1,101)},n_jobs =-1,verbose = 0):
  ''' function for best random forest model using gridsearch ,X_train - train features, X_test - test features, y_train - train target, y_test - test target, metric - metric to use,k - number of validation set,max_samples - how much of the data the tree uses,  params1 - parameters to use for the first grid search,params2 - parameters to use for the second grid search,n_jobs - number of cpu used,verbose - verbose
  '''
  if verbose != 0:
    n_jobs = 1
  kf = KFold(n_splits=k, random_state=42, shuffle=True)
  model = RandomForestClassifier(n_estimators= 25, random_state = 42)
  grid_search = GridSearchCV(model, param_grid=params1, cv=kf, scoring=metric,return_train_score= True,n_jobs=n_jobs,verbose = verbose)
  grid_search = grid_search.fit(X_train,y_train)
  best_max_depth = grid_search.best_params_["max_depth"]
  best_min_samples_leaf = grid_search.best_params_["min_samples_leaf"]
  model = RandomForestClassifier(max_depth=best_max_depth,min_samples_leaf =best_min_samples_leaf,random_state = 42)
  grid_search = GridSearchCV(model, param_grid=params2, cv=kf, scoring=metric,return_train_score= True,n_jobs=n_jobs, verbose = verbose)
  grid_search = grid_search.fit(X_train,y_train)
  best_n_estimators = grid_search.best_params_["n_estimators"]
  model = RandomForestClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf,n_estimators = best_n_estimators,random_state = 42)
  model.fit(X_train,y_train)
  grid = pd.DataFrame(grid_search.cv_results_)
  best_train_score = grid.loc[grid_search.best_index_, "mean_train_score"]
  PRINT(f"max_depth = {best_max_depth}, min_samples_leaf = {best_min_samples_leaf},n_estimators = {best_n_estimators} gives the best random forest model with a {metric} of {grid_search.best_score_} on the test(validation) set and a {metric} of {best_train_score} on the training set")
  return model ,grid
def svm(X_train,X_test,y_train,y_test,metric,k =4 , params = {"kernel": ["linear", "poly","rbf", "sigmoid"],"nu":np.arange(0.01,1,0.01)},n_jobs = -1,verbose = 0):
  ''' function for best svm model using gridsearch,X_train - train features, X_test - test features, y_train - train target, y_test - test target, metric - metric to use,k - number of validation set, params - parameters to use,n_jobs - number of cpu used,verbose - verbose
  '''
  if verbose != 0:
    n_jobs = 1
  kf = KFold(n_splits=k, random_state=42, shuffle=True)
  model = NuSVC()
  grid_search = GridSearchCV(model, param_grid=params, cv=kf, scoring=metric,return_train_score= True,n_jobs=n_jobs,verbose = verbose)
  grid_search = grid_search.fit(X_train,y_train)
  best_kernel = grid_search.best_params_["kernel"]
  best_nu = grid_search.best_params_["nu"]
  model = NuSVC(kernel=best_kernel, nu=best_nu)
  model.fit(X_train,y_train)
  grid = pd.DataFrame(grid_search.cv_results_)
  best_train_score = grid.loc[grid_search.best_index_, "mean_train_score"]
  PRINT(f"kernel = {best_kernel}, nu = {best_nu} gives the best svm model with a {metric} of {grid_search.best_score_} on the test set and a {metric} of {best_train_score} on the training set")
  return model ,grid
def make_model_and_evaluate(model,X_train,X_test,y_train,y_test,metric,vebrose= 0 ):
  if model == "ncc":
    return ncc(X_train,X_test,y_train,y_test,metric)
  elif model == "knn":
    return knn(X_train,X_test,y_train,y_test,metric,vebrose)
  elif model == "tree":
    return tree(X_train,X_test,y_train,y_test,metric,vebrose)
  elif model == "random forest":
    return random_forest(X_train,X_test,y_train,y_test,metric,vebrose)
  elif model == "svm":
    return svm(X_train,X_test,y_train,y_test,metric,vebrose)
#DUPLICATES
def duplicates(df):
  return f"there are {df.duplicated().sum()} duplicates"
def removeduplicates(df,copy = False):
  if copy:
    df1 = df.copy()
    df1.drop_duplicates(inplace=True)
    return df1
  df.drop_duplicates(inplace=True)
  return df
def final_evaluation(model,X_train,X_test,y_train ,y_test,metric):
  y_pred = model.predict(X_test)
  y_train_pred = model.predict(X_train)
  if metric == "accuracy":
    score = accuracy_score(y_test,y_pred)
    train_score = accuracy_score(y_train,y_train_pred)
    return(f"the final {metric} on the test set is {score*100}% and on the training set is {train_score*100}%")
  elif metric == "recall":
    score = recall_score(y_test,y_pred)
    train_score = recall_score(y_train,y_train_pred)
    return(f"the final {metric} on the test set is {score*100}% and on the training set is {train_score*100}%")
  elif metric == "precision":
    score = precision_score(y_test,y_pred)
    train_score = precision_score(y_train,y_train_pred)
    return(f"the final {metric} on the test set is {score*100}% and on the training set is {train_score*100}%")
  elif metric == "f1":
    score = f1_score(y_test,y_pred)
    train_score = f1_score(y_train,y_train_pred)
    return(f"the final {metric} on the test set is {score*100}% and on the training set is {train_score*100}%")