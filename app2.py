import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, validation_curve, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import functions
# 1. Create your data
st.set_page_config(
    page_title="AI Model Maker",
    page_icon="🤖",
    layout="wide"
)
def load_data():
    st.title("AI")
    link = st.text_input("Enter link:")
    if st.button("Load CSV"):
        if link == "":
            st.warning("Please enter a link to load the data.") 
        else:
            try:
                st.session_state.df = pd.read_csv(link)
                st.session_state.outliners_dict = {"column": [],"count": []}
                for i in st.session_state.df.select_dtypes(include=np.number).columns:
                    text , count = functions.outliners(st.session_state.df,i)
                    if count > 0:
                        st.session_state.outliners_dict["column"].append(i)
                        st.session_state.outliners_dict["count"].append(count)
                st.dataframe(st.session_state.df, width='stretch')
            except Exception as e:
                st.warning(f"Error loading CSV: {e}")
def EDA():
    if "df" not in st.session_state:
        st.info("No data found. Please load a CSV on the 'Load Data' page first.")
    else:
        df = st.session_state.df
        st.title("EDA")
        func = st.selectbox("Select EDA function", options=["show data" , "describe", "info" ,"nulls" , "outliers" , "correlation" , "duplicates"], key="eda_function")
        if func == "describe":
            st.write(df.describe())
        elif func == "info":
            info_df = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.values.astype(str),
                    "Non-Null Count": df.notnull().sum().values.astype(str)
                })
            st.dataframe(info_df, width='stretch')
            st.write(f"**Total Rows:** {df.shape[0]} | **Total Columns:** {df.shape[1]}")
        elif func == "show data":
            st.dataframe(df, width='stretch')
        elif func == "nulls":
            st.write(functions.null(df))
        elif func == "outliers":
            st.session_state.outliners_dict = {"column": [],"count": []}
            for i in df.select_dtypes(include=np.number).columns:
                text , count = functions.outliners(df,i)
                if count > 0:
                    st.session_state.outliners_dict["column"].append(i)
                    st.session_state.outliners_dict["count"].append(count)
                st.write(text)
        elif func == "correlation":
            numeric_df = df.select_dtypes(include=['number'])
            corr = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title("Correlation Heatmap")
            st.pyplot(plt)
        elif func == "duplicates":
            st.write(functions.duplicates(df))
def plots():
    st.title("Plot")
    
    # Safety check: ensure data is loaded in session state
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # 1. Main Plot Selection - 'key' makes this choice stay saved
        func = st.selectbox(
            "Select Plot Type", 
            options=["histogram", "boxplot", "scatter", "count", "line", "bar", "heatmap"], 
            key="main_plot_selection"
        )

        # 2. Create the Figure and Axis (the 'canvas')
        fig, ax = plt.subplots(figsize=(10, 6))

        # 3. Plot Logic
        if func == "histogram":
            column = st.selectbox("Select column", options=df.columns, key="hist_col")
            sns.histplot(df[column], kde=True, ax=ax)
            ax.set_title(f"Histogram of {column}")

        elif func == "boxplot":
            column = st.selectbox("Select column", options=df.columns, key="box_col")
            sns.boxplot(x=df[column], ax=ax)
            ax.set_title(f"Boxplot of {column}")

        elif func == "scatter":
            x_col = st.selectbox("Select X-axis", options=df.columns, key="scat_x")
            y_col = st.selectbox("Select Y-axis", options=df.columns, key="scat_y")
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
            ax.set_title(f"Scatter: {x_col} vs {y_col}")

        elif func == "count":
            column = st.selectbox("Select column", options=df.columns, key="count_col")
            sns.countplot(x=df[column], ax=ax)
            ax.set_title(f"Count of {column}")

        elif func == "line":
            x_col = st.selectbox("Select X-axis", options=df.columns, key="line_x")
            y_col = st.selectbox("Select Y-axis", options=df.columns, key="line_y")
            sns.lineplot(x=df[x_col], y=df[y_col], ax=ax)
            ax.set_title(f"Line Plot: {x_col} vs {y_col}")

        elif func == "bar":
            column = st.selectbox("Select column", options=df.columns, key="bar_col")
            # Calculate counts first to avoid Seaborn index errors
            counts = df[column].value_counts()
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
            ax.set_title(f"Bar Plot of {column}")

        elif func == "heatmap":
            # Only use numeric columns for correlation
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title("Correlation Heatmap")
            else:
                st.warning("Heatmap requires at least one numeric column.")

        # 4. Display the plot in the app
        st.pyplot(fig)
        
    else:
        st.info("No data found. Please load a CSV on the 'Load Data' page first.")
def preprocessing():
    if "df" in st.session_state:
        df = st.session_state.df
        if "encode_method" not in st.session_state:
            st.session_state.encode_method = None
        if "column_to_encode" not in st.session_state:
            st.session_state.column_to_encode = []
        if "scaling_method" not in st.session_state:
            st.session_state.scaling_method = None
        user = st.session_state["username"]
        st.title("Preprocessing")
        
        func = st.selectbox("Select Preprocessing function", 
                            options=["nulls","outliers","duplicates", "encode", "scaling" , "feature_engineering" ,"drop"], 
                            key="preprocess_function")

        if func == "nulls":
            null_counts = df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0].index.tolist()

            if not cols_with_nulls:
                st.success("🎉 Your dataset is clean! No missing values found.")
            else:
                col_to_fix = st.selectbox("Choose a column to fix:", cols_with_nulls, key="null_col_selector")
                st.write(f"Column `{col_to_fix}` has **{null_counts[col_to_fix]}** missing values.")

                method = st.radio("Choose a strategy:", 
                                ["Do Nothing", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode" ,"fill with uniform"], 
                                key="null_method_radio")

                if st.button("Apply Fix to Dataset"):
                    if method == "Drop Rows":
                        st.session_state.df = df.dropna(subset=[col_to_fix])
                    elif method == "Fill with Mean":
                        if pd.api.types.is_numeric_dtype(df[col_to_fix]):
                            val = df[col_to_fix].mean()
                            st.session_state.df[col_to_fix] = df[col_to_fix].fillna(val)
                        else:
                            st.error("Mean only works for numeric columns!")
                    elif method == "Fill with Median":
                        if pd.api.types.is_numeric_dtype(df[col_to_fix]):
                            val = df[col_to_fix].median()
                            st.session_state.df[col_to_fix] = df[col_to_fix].fillna(val)
                    elif method == "Fill with Mode":
                        val = df[col_to_fix].mode()[0]
                        st.session_state.df[col_to_fix] = df[col_to_fix].fillna(val)
                    elif method == "fill with uniform":
                        st.session_state.df =  functions.fill_uniform(st.session_state.df, col_to_fix)
                    
                    st.rerun()

        elif func == "outliers":
            outliners_dict = st.session_state.outliners_dict
            column = st.selectbox("Select column to remove outliers", options=list(outliners_dict["column"]), key="outlier_col")
            st.write(f"Number of outliers in {column}: {outliners_dict['count'][outliners_dict['column'].index(column)]}")
            if st.button("Remove Outliers"):
                # Update session state directly
                st.session_state.df = functions.removeoutliners(st.session_state.df, column)
                st.rerun()
                
            if st.button("squash outliers"):
                st.session_state.df = functions.squashoutliners(st.session_state.df, column)
                st.rerun()
    
        elif func == "duplicates":
            st.write(functions.duplicates(df))
            if st.button("Remove Duplicates"):
                st.session_state.df = functions.removeduplicates(df)
                st.rerun()
        elif func == "drop":
            st.subheader("🗑️ Drop Rows or Columns")
            drop_mode = st.radio("What would you like to drop?", ["By Query (Rows)", "Specific Columns"], key="drop_mode")

            if drop_mode == "By Query (Rows)":
                st.markdown("""
                **Query Examples:**
                * `Age > 30` (Removes rows where Age is greater than 30)
                * `Status == 'Inactive'` (Removes rows where Status is Inactive)
                """)
                query_str = st.text_input("Enter query to drop rows:", placeholder="example: Price > 100", key="drop_query_input")
                
                if st.button("Drop Rows by Query"):
                    try:
                        # We get the indices of the rows that match the query
                        to_drop = df.query(query_str).index
                        st.session_state.df = df.drop(index=to_drop)
                        st.success(f"Dropped {len(to_drop)} rows.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error in query: {e}")

            elif drop_mode == "Specific Columns":
                cols_to_drop = st.multiselect("Select columns to remove:", options=df.columns, key="drop_col_select")
                
                if st.button("Drop Selected Columns"):
                    if cols_to_drop:
                        st.session_state.df = df.drop(columns=cols_to_drop)
                        st.success(f"Dropped columns: {', '.join(cols_to_drop)}")
                        st.rerun()
                    else:
                        st.warning("Please select at least one column.")
        elif func == "encode":
            column = st.selectbox("Select column", options=st.session_state.df.select_dtypes(include=['object']).columns)
            method = st.selectbox("Select method", options=["Label Encoding", "One-Hot Encoding"])
            if st.button("Encode"):
                st.session_state.column_to_encode.append(column)
                if method == "Label Encoding":
                    st.session_state.df, st.session_state.encoder = functions.label_encoding(st.session_state.df, column)
                else:
                    st.session_state.df, st.session_state.encoder = functions.one_hot_encoding(st.session_state.df, [column])
                st.success(f"Encoded {column}!")
            st.session_state.encode_method = method
            user = st.session_state["username"]
            user_path = f"saved_encoding_method/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'encode_method' in st.session_state:
                file_path = f"{user_path}/encoding_method.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.encode_method, f)
            user_path = f"saved_encoded_columns/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'column_to_encode' in st.session_state:
                file_path = f"{user_path}/encoded_columns.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.column_to_encode, f)
        elif func == "scaling":
            column   = st.multiselect("Select column to scale", options=df.select_dtypes(include=np.number).columns, key="scale_col")
            method = st.selectbox("Select scaling method", options=["Standard Scaling", "Min-Max Scaling"], key="scale_method")
            st.session_state.scaling_method = None
            if st.button("Scale"):
                st.session_state.DF_BEFORE_SCALING = df.copy()
                st.session_state.columns_to_scale = column
                if method == "Standard Scaling":
                    st.session_state.df , st.session_state.scaler = functions.standardscaler(df, column)
                elif method == "Min-Max Scaling":
                    st.session_state.df   , st.session_state.scaler = functions.minmaxscalar(df, column)
                st.rerun()
            st.session_state.scaling_method = method
            user_path = f"saved_scaling_method/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'scaling_method' in st.session_state:
                file_path = f"{user_path}/scaling_method.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.scaling_method, f)
            user_path = f"saved_scaled_columns/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'columns_to_scale' in st.session_state:
                file_path = f"{user_path}/scaled_columns.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.columns_to_scale, f)
        elif func == "feature_engineering":
            st.write("Feature engineering functionality coming soon!")
        st.session_state.df =  st.session_state.df.copy()
        st.dataframe(st.session_state.df, width='stretch')
        st.write(f"Total Rows: {st.session_state.df.shape[0]} | Total Columns: {st.session_state.df.shape[1]}")
    else:
        st.info("No data found. Please load a CSV on the 'Load Data' page first.")
def split_data():
    st.title("Split Data")
    if "df" in st.session_state:
        df = st.session_state.df
        
        y_column = st.selectbox("Select target column", options=df.columns, key="model_target_col")
        x_columns = st.multiselect("Select feature columns", options=[col for col in df.columns if col != y_column], key="model_feature_cols")
        st.session_state.target = y_column
        st.session_state.features = x_columns
        user = st.session_state["username"]
        test_size = st.slider("Select test size", 0.1, 0.9, 0.2, key="train_size_slider")
        if st.button("Initialize / Reset Split"):
            y = df[y_column]
            X = df[x_columns]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            user_path = f"saved_features/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            file_path = f"{user_path}/features.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(st.session_state.features, f)
            if "DF_BEFORE_SCALING" in st.session_state:
                st.session_state.X_test_before_scaling = st.session_state.DF_BEFORE_SCALING[x_columns].loc[X_test.index]
            else: 
                st.session_state.X_test_before_scaling = st.session_state.X_test
            user_path = f"saved_X_test_before_scaling/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'X_test_before_scaling' in st.session_state:
                file_path = f"{user_path}/X_test_before_scaling.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.X_test_before_scaling, f)
            st.success("Data split successfully!")

        if "X_train" in st.session_state:
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Apply SMOTE"):
                    st.session_state.X_train, st.session_state.y_train = functions.oversampling(st.session_state.X_train, st.session_state.y_train)
                    st.success("Balanced with SMOTE")

            with col2:
                if st.button("Apply Undersampling"):
                    st.session_state.X_train, st.session_state.y_train = functions.undersampling(st.session_state.X_train, st.session_state.y_train)
                    st.success("Balanced with Undersampling")
            st.divider()
            st.subheader("Class Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.countplot(x=st.session_state.y_train, ax=ax)
            st.pyplot(fig)
    else:
        st.info("No data found. Please load a CSV on the 'Load Data' page first.")

def make_model():
    user = st.session_state["username"]
    user_path = f"saved_x_test/{user}"
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    if 'X_test' in st.session_state:
        file_path = f"{user_path}/x_test.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(st.session_state.X_test, f)
    user_path = f"saved_x_train/{user}"
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    if 'X_train' in st.session_state:
        file_path = f"{user_path}/x_train.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(st.session_state.X_train, f)
    user_path = f"saved_y_test/{user}"
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    if 'y_test' in st.session_state:
        file_path = f"{user_path}/y_test.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(st.session_state.y_test, f)
    user_path = f"saved_y_train/{user}"
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    if 'y_train' in st.session_state:
        file_path = f"{user_path}/y_train.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(st.session_state.y_train, f)
            
    st.title("Make Model")
    if "X_train" not in st.session_state:
        st.info("No training data found. Please split your data on the 'Split Data' page first.")
        return
    model_name = st.selectbox("Select model", options=["KNN", "Decision Tree", "Random Forest", "SVM", "Nearest Centroid"], key="model_select")
    if model_name == "KNN":
        metrics = st.multiselect("Select metrics", options=["accuracy", "precision", "recall", "f1"], key="knn_metrics")
        if 'knn_mode' not in st.session_state:
            st.session_state.knn_mode = None
        col1, col2 = st.columns(2)
        if col1.button("Manual Entry"):
            st.session_state.knn_mode = "manual"
        if col2.button("Grid Search"):
            st.session_state.knn_mode = "grid"
        if st.button("Save Model to My Account", key="knn_save"):
            user = st.session_state["username"]
            user_path = f"saved_models/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'knn_model' in st.session_state:
                file_path = f"{user_path}/knn_model.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.knn_model, f)
                st.success(f"Model saved in your folder: {file_path}")
            else:
                st.warning("Please train a model first before saving.")
        if st.session_state.knn_mode == "manual":
            st.subheader("Manual")
            with st.form("knn_manual_form"):
                k = st.number_input("Number of neighbors (k)", min_value=1, value=5, step=1)
                submit_manual = st.form_submit_button("Train KNN Model")
                
                if submit_manual:
                    model = KNeighborsClassifier(n_neighbors=int(k))
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.knn_model = model
                    
                    st.success(f"KNN trained with k={k}")
                    for metric in metrics:
                        st.text(functions.final_evaluation(
                            st.session_state.knn_model,
                            st.session_state.X_train, st.session_state.X_test,
                            st.session_state.y_train, st.session_state.y_test,
                            metric
                        ))

        elif st.session_state.knn_mode == "grid":
            st.subheader("GridSearchCV")
            verbose = st.slider("Verbose (you will see the progress in the terminal)", 0, 12, 11, key="knn_verbose")
            param_grid = {
                'n_neighbors': np.arange(1 , 100 , 1)
            }
            prams =st.text_input("Range for n_neighbors (default: 1,100,1)",placeholder="example:1 , 10 , 2 (from 1 to 10 with jumps of 2)", key="knn_param_input").split(',')
            start= int(prams[0]) if len(prams) > 0 and prams[0].strip().isdigit() else 1
            end = int(prams[1]) + 1 if len(prams) > 1 and prams[1].strip().isdigit() else 101
            step = int(prams[2]) if len(prams) > 2 and prams[2].strip().isdigit() else 1
            param_grid['n_neighbors'] = np.arange(start, end, step)
            k = st.text_input("Number of folds for K-Fold (default: 4) (if you dont want CV put 0)", key="knn_kfold_input")
            n_splits = int(k) if k.isdigit() else 4
            metric = st.selectbox("Select ONE metric for evaluation during GridSearch", options=["accuracy", "precision", "recall", "f1"], key="knn_grid_metrics")
            if st.button("Run Search"):
                st.info("Running GridSearch...")
                st.session_state.knn_model , st.session_state.knn_grid = functions.knn(X_train=st.session_state.X_train, y_train=st.session_state.y_train,X_test = st.session_state.X_test,y_test = st.session_state.y_test , metric = metric , k= n_splits , params=param_grid , verbose=verbose)
                st.success("GridSearch completed!")
                st.text(f"Best k: {st.session_state.knn_model.n_neighbors}")
                for metric in metrics:
                    st.text(functions.final_evaluation(
                        st.session_state.knn_model,
                        st.session_state.X_train, st.session_state.X_test,
                        st.session_state.y_train, st.session_state.y_test,
                        metric
                    ))
            if st.button("Show GridSearchCV Results"):
                if 'knn_grid' in st.session_state:
                    grid_results = pd.DataFrame(st.session_state.knn_grid)
                    st.dataframe(grid_results, width='stretch')
                else:
                    st.warning("Please run the GridSearch first to see results.")
            if st.button("Show Validation Curve for n_neighbors"):
                if 'knn_grid' in st.session_state:
                    param_range = param_grid['n_neighbors']
                    train_scores, test_scores = validation_curve(
                        KNeighborsClassifier(),
                        st.session_state.X_train,
                        st.session_state.y_train,
                        param_name="n_neighbors",
                        param_range=param_range,
                        cv=n_splits,
                        scoring=metric
                    )
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(param_range, train_scores_mean, label='Training score', color='blue')
                    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='orange')
                    plt.title('Validation Curve for KNN')
                    plt.xlabel('Number of Neighbors (k)')
                    plt.ylabel(metric.capitalize())
                    plt.legend(loc='best')
                    st.pyplot(plt)
                else:
                    st.warning("Please run the GridSearch first to see the validation curve.")
    elif model_name == "Decision Tree":
        metrics = st.multiselect("Select metrics", options=["accuracy", "precision", "recall", "f1"], key="dt_metrics")
        if 'dt_mode' not in st.session_state:
            st.session_state.dt_mode = None
        col1, col2 = st.columns(2)
        if col1.button("Manual Entry"):
            st.session_state.dt_mode = "manual"
        if col2.button("Grid Search"):
            st.session_state.dt_mode = "grid"
        if st.button("Save Model to My Account", key="dt_save"):
            user = st.session_state["username"]
            user_path = f"saved_models/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'dt_model' in st.session_state:
                file_path = f"{user_path}/decision_tree_model.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.dt_model, f)
                st.success(f"Model saved in your folder: {file_path}")
            else:
                st.warning("Please train a model first before saving.")
        if st.session_state.dt_mode == "manual":
            st.subheader("Manual")
            with st.form("dt_manual_form"):
                max_depth = st.number_input("Maximum depth of the tree", min_value=1, value=5, step=1)
                min_samples_leaf = st.number_input("Minimum samples in a leaf node", min_value=1, value=1, step=1)
                submit_manual = st.form_submit_button("Train Decision Tree Model")

                if submit_manual:
                    model = DecisionTreeClassifier(max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf))
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.dt_model = model

                    st.success(f"Decision Tree trained with max_depth={max_depth} and min_samples_leaf={min_samples_leaf}")
                    for metric in metrics:
                        st.text(functions.final_evaluation(
                            st.session_state.dt_model,
                            st.session_state.X_train, st.session_state.X_test,
                            st.session_state.y_train, st.session_state.y_test,
                            metric
                        ))
        elif st.session_state.dt_mode == "grid":
            st.subheader("GridSearchCV")
            verbose = st.slider("Verbose (you will see the progress in the terminal)", 0, 12, 1, key="dt_verbose")
            param_grid = {
                'max_depth': np.arange(1, 20, 1),
                'min_samples_leaf': np.arange(1, 20, 1)
            }
            max_depth_range = st.text_input("Range for max_depth (default: 1,20,1)", placeholder="example:1 , 20 , 1 (from 1 to 20 with jumps of 1)", key="dt_param_input").split(',')
            min_samples_leaf_range = st.text_input("Range for min_samples_leaf (default: 1,20,1)", placeholder="example:1 , 20 , 1 (from 1 to 20 with jumps of 1)", key="dt_min_samples_leaf_input").split(',') 
            min_samples_leaf_start = int(min_samples_leaf_range[0]) if len(min_samples_leaf_range) > 0 and min_samples_leaf_range[0].strip().isdigit() else 1
            min_samples_leaf_end = int(min_samples_leaf_range[1]) + 1 if len(min_samples_leaf_range) > 1 and min_samples_leaf_range[1].strip().isdigit() else 21
            min_samples_leaf_step = int(min_samples_leaf_range[2]) if len(min_samples_leaf_range) > 2 and min_samples_leaf_range[2].strip().isdigit() else 1
            param_grid['min_samples_leaf'] = np.arange(min_samples_leaf_start, min_samples_leaf_end, min_samples_leaf_step)
            max_depth_start = int(max_depth_range[0]) if len(max_depth_range) > 0 and max_depth_range[0].strip().isdigit() else 1
            max_depth_end = int(max_depth_range[1]) + 1 if len(max_depth_range) > 1 and max_depth_range[1].strip().isdigit() else 21
            step = int(max_depth_range[2]) if len(max_depth_range) > 2 and max_depth_range[2].strip().isdigit() else 1
            param_grid['max_depth'] = np.arange(max_depth_start, max_depth_end, step)
            k = st.text_input("Number of folds for K-Fold (default: 4) (if you dont want CV put 0)", key="dt_kfold_input")
            n_splits = int(k) if k.isdigit() else 4
            metric = st.selectbox("Select ONE metric for evaluation during GridSearch", options=["accuracy", "precision", "recall", "f1"], key="dt_grid_metrics")
            if st.button("Run Search"):
                st.info("Running GridSearch...")
                st.session_state.dt_model , st.session_state.dt_grid = functions.tree(X_train=st.session_state.X_train, y_train=st.session_state.y_train,X_test = st.session_state.X_test,y_test = st.session_state.y_test , metric = metric , k= n_splits , params=param_grid , verbose=verbose)
                st.success("GridSearch completed!")
                st.text(f"Best max_depth: {st.session_state.dt_model.max_depth}")
                st.text(f"Best min_samples_leaf: {st.session_state.dt_model.min_samples_leaf}")
                for metric in metrics:
                    st.text(functions.final_evaluation(
                        st.session_state.dt_model,
                        st.session_state.X_train, st.session_state.X_test,
                        st.session_state.y_train, st.session_state.y_test,
                        metric
                    ))
            if st.button("Show GridSearchCV Results"):
                if 'dt_grid' in st.session_state:
                    grid_results = pd.DataFrame(st.session_state.dt_grid)
                    st.dataframe(grid_results, width='stretch')
                else:
                    st.warning("Please run the GridSearch first to see results.")
            if st.button("Show Validation Curve for max_depth"):
                if 'dt_grid' in st.session_state:
                    param_range = param_grid['max_depth']
                    train_scores, test_scores = validation_curve(
                        DecisionTreeClassifier(),
                        st.session_state.X_train,
                        st.session_state.y_train,
                        param_name="max_depth",
                        param_range=param_range,
                        cv=n_splits,
                        scoring=metric
                    )
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(param_range, train_scores_mean, label='Training score', color='blue')
                    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='orange')
                    plt.title('Validation Curve for Decision Tree')
                    plt.xlabel('Maximum Depth of Tree')
                    plt.ylabel(metric.capitalize())
                    plt.legend(loc='best')
                    st.pyplot(plt)
                else:
                    st.warning("Please run the GridSearch first to see the validation curve.")
            if st.button("Show Validation Curve for min_samples_leaf"):
                if 'dt_grid' in st.session_state:
                    param_range = param_grid['min_samples_leaf']
                    train_scores, test_scores = validation_curve(
                        DecisionTreeClassifier(),
                        st.session_state.X_train,
                        st.session_state.y_train,
                        param_name="min_samples_leaf",
                        param_range=param_range,
                        cv=n_splits,
                        scoring=metric
                    )
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(param_range, train_scores_mean, label='Training score', color='blue')
                    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='orange')
                    plt.title('Validation Curve for min_samples_leaf')
                    plt.xlabel('Minimum Samples in Leaf')
                    plt.ylabel(metric.capitalize())
                    plt.legend(loc='best')
                    st.pyplot(plt)
    elif  model_name == "Random Forest":
        metrics = st.multiselect("Select metrics", options=["accuracy", "precision", "recall", "f1"], key="rf_metrics")
        if 'rf_mode' not in st.session_state:
            st.session_state.rf_mode = None
        col1, col2 = st.columns(2)
        if col1.button("Manual Entry"):
            st.session_state.rf_mode = "manual"
        if col2.button("Grid Search"):
            st.session_state.rf_mode = "grid"
        if st.button("Save Model to My Account", key="rf_save"):
            user = st.session_state["username"]
            user_path = f"saved_models/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'rf_model' in st.session_state:
                file_path = f"{user_path}/random_forest_model.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.rf_model, f)
                st.success(f"Model saved in your folder: {file_path}")
            else:
                st.warning("Please train a model first before saving.")
        if st.session_state.rf_mode == "manual":
            st.subheader("Manual")
            with st.form("rf_manual_form"):
                n_estimators = st.number_input("Number of trees in the forest", min_value=1, value=100, step=1)
                max_depth = st.number_input("Maximum depth of the tree", min_value=1, value=5, step=1)
                min_samples_leaf = st.number_input("Minimum samples in a leaf node", min_value=1, value=1, step=1)
                submit_manual = st.form_submit_button("Train Random Forest Model")

                if submit_manual:
                    model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf))
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.rf_model = model
                    st.success(f"Random Forest trained with n_estimators={n_estimators} and max_depth={max_depth} , min_samples_leaf={min_samples_leaf}")
                    for metric in metrics:
                        st.text(functions.final_evaluation(
                            st.session_state.rf_model,
                            st.session_state.X_train, st.session_state.X_test,
                            st.session_state.y_train, st.session_state.y_test,
                            metric
                        ))
        elif st.session_state.rf_mode == "grid":
            st.subheader("GridSearchCV")
            verbose = st.slider("Verbose (you will see the progress in the terminal)", 0, 12, 1, key="rf_verbose")
            params1 = {
                'max_depth': np.arange(1, 101, 2),
                'min_samples_leaf': np.arange(1, 51, 1)
            }
            params2 = {     
                'n_estimators': np.arange(10, 101, 10)
            }
            n_estimators_range = st.text_input("Range for n_estimators (default: 10,100,10)", placeholder="example:10 , 100 , 10 (from 10 to 100 with jumps of 10)", key="rf_n_estimators_input").split(',')
            max_depth_range = st.text_input("Range for max_depth (default: 1,100,10)", placeholder="example:1 , 100 , 1 (from 1 to 100 with jumps of 1)", key="rf_max_depth_input").split(',')
            min_samples_leaf_range = st.text_input("Range for min_samples_leaf (default: 1,50,1)", placeholder="example:1 , 50 , 1 (from 1 to 50 with jumps of 1)", key="rf_min_samples_leaf_input").split(',') 
            
            n_estimators_start = int(n_estimators_range[0]) if len(n_estimators_range) > 0 and n_estimators_range[0].strip().isdigit() else 10
            n_estimators_end = int(n_estimators_range[1]) + 1 if len(n_estimators_range) > 1 and n_estimators_range[1].strip().isdigit() else 101
            n_estimators_step = int(n_estimators_range[2]) if len(n_estimators_range) > 2 and n_estimators_range[2].strip().isdigit() else 10
            params2['n_estimators'] = np.arange(n_estimators_start, n_estimators_end, n_estimators_step)

            max_depth_start = int(max_depth_range[0]) if len(max_depth_range) > 0 and max_depth_range[0].strip().isdigit() else 1
            max_depth_end = int(max_depth_range[1]) + 1 if len(max_depth_range) > 1 and max_depth_range[1].strip().isdigit() else 101
            max_depth_step = int(max_depth_range[2]) if len(max_depth_range) > 2 and max_depth_range[2].strip().isdigit() else 10
            params1['max_depth'] = np.arange(max_depth_start, max_depth_end, max_depth_step)

            min_samples_leaf_start = int(min_samples_leaf_range[0]) if len(min_samples_leaf_range) > 0 and min_samples_leaf_range[0].strip().isdigit() else 1
            min_samples_leaf_end = int(min_samples_leaf_range[1]) + 1 if len(min_samples_leaf_range) > 1 and min_samples_leaf_range[1].strip().isdigit() else 51
            min_samples_leaf_step = int(min_samples_leaf_range[2]) if len(min_samples_leaf_range) > 2 and min_samples_leaf_range[2].strip().isdigit() else 1
            params1['min_samples_leaf'] = np.arange(min_samples_leaf_start, min_samples_leaf_end, min_samples_leaf_step)
            k = st.text_input("Number of folds for K-Fold (default: 4) (if you dont want CV put 0)", key="rf_kfold_input")
            n_splits = int(k) if k.isdigit() else 4
            metric = st.selectbox("Select ONE metric for evaluation during GridSearch", options=["accuracy", "precision", "recall", "f1"], key="rf_grid_metrics")
            if st.button("Run Search"):
                st.info("Running GridSearch...")
                st.session_state.rf_model , st.session_state.rf_grid = functions.random_forest(X_train=st.session_state.X_train, y_train=st.session_state.y_train,X_test = st.session_state.X_test,y_test = st.session_state.y_test , metric = metric , k= n_splits , params1=params1, params2=params2 , verbose=verbose)
                st.success("GridSearch completed!")
                st.text(f"Best n_estimators: {st.session_state.rf_model.n_estimators}")
                st.text(f"Best max_depth: {st.session_state.rf_model.max_depth}")
                st.text(f"Best min_samples_leaf: {st.session_state.rf_model.min_samples_leaf}")
                for metric in metrics:
                    st.text(functions.final_evaluation(
                        st.session_state.rf_model,
                        st.session_state.X_train, st.session_state.X_test,
                        st.session_state.y_train, st.session_state.y_test,
                        metric
                    ))
            if st.button("Show GridSearchCV Results"):
                if 'rf_grid' in st.session_state:
                    grid_results = pd.DataFrame(st.session_state.rf_grid)
                    st.dataframe(grid_results, width='stretch')
                else:
                    st.warning("Please run the GridSearch first to see results.")
            if st.button("Show Validation Curve for n_estimators"):
                if 'rf_grid' in st.session_state:
                    param_range = params2['n_estimators']
                    train_scores, test_scores = validation_curve(
                        RandomForestClassifier(max_depth=st.session_state.rf_model.max_depth, min_samples_leaf=st.session_state.rf_model.min_samples_leaf),
                        st.session_state.X_train,
                        st.session_state.y_train,
                        param_name="n_estimators",
                        param_range=param_range,
                        cv=n_splits,
                        scoring=metric
                    )
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(param_range, train_scores_mean, label='Training score', color='blue')
                    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='orange')
                    plt.title('Validation Curve for n_estimators')
                    plt.xlabel('Number of Trees (n_estimators)')
                    plt.ylabel(metric.capitalize())
                    plt.legend(loc='best')
                    st.pyplot(plt)
                else:
                    st.warning("Please run the GridSearch first to see the validation curve.")
            if st.button("Show Validation Curve for max_depth"):
                if 'rf_grid' in st.session_state:
                    param_range = params1['max_depth']
                    train_scores, test_scores = validation_curve(
                        RandomForestClassifier(n_estimators=st.session_state.rf_model.n_estimators, min_samples_leaf=st.session_state.rf_model.min_samples_leaf),
                        st.session_state.X_train,
                        st.session_state.y_train,
                        param_name="max_depth",
                        param_range=param_range,
                        cv=n_splits,
                        scoring=metric
                    )
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(param_range, train_scores_mean, label='Training score', color='blue')
                    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='orange')
                    plt.title('Validation Curve for max_depth')
                    plt.xlabel('Maximum Depth of Tree')
                    plt.ylabel(metric.capitalize())
                    plt.legend(loc='best')
                    st.pyplot(plt)
                else:
                    st.warning("Please run the GridSearch first to see the validation curve.")
            if st.button("Show Validation Curve for min_samples_leaf"):
                if 'rf_grid' in st.session_state:
                    param_range = params1['min_samples_leaf']
                    train_scores, test_scores = validation_curve(
                        RandomForestClassifier(n_estimators=st.session_state.rf_model.n_estimators, max_depth=st.session_state.rf_model.max_depth),
                        st.session_state.X_train,
                        st.session_state.y_train,
                        param_name="min_samples_leaf",
                        param_range=param_range,
                        cv=n_splits,
                        scoring=metric
                    )
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(param_range, train_scores_mean, label='Training score', color='blue')
                    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='orange')
                    plt.title('Validation Curve for min_samples_leaf')
                    plt.xlabel('Minimum Samples in Leaf')
                    plt.ylabel(metric.capitalize())
                    plt.legend(loc='best')
                    st.pyplot(plt)
                else:
                    st.warning("Please run the GridSearch first to see the validation curve.")
    elif model_name == "SVM":
        metrics = st.multiselect("Select metrics", options=["accuracy", "precision", "recall", "f1"], key="svm_metrics")
        if 'svm_mode' not in st.session_state:
            st.session_state.svm_mode = None
        col1, col2 = st.columns(2)
        if col1.button("Manual Entry"):
            st.session_state.svm_mode = "manual"
        if col2.button("Grid Search"):
            st.session_state.svm_mode = "grid"
        if st.button("Save Model to My Account", key="svm_save"):
            user = st.session_state["username"]
            user_path = f"saved_models/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'svm_model' in st.session_state:
                file_path = f"{user_path}/svm_model.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.svm_model, f)
                st.success(f"Model saved in your folder: {file_path}")
            else:
                st.warning("Please train a model first before saving.")
        if st.session_state.svm_mode == "manual":
            st.subheader("Manual")
            with st.form("svm_manual_form"):
                nu = st.number_input("Nu (upper bound on training errors, between 0 and 1)", min_value=0.01, max_value=1.0, value=0.5, step=0.05)
                kernel = st.selectbox("Kernel", options=["linear", "poly", "rbf", "sigmoid"], key="svm_kernel")
                submit_manual = st.form_submit_button("Train Nu-SVM Model")

                if submit_manual:
                    model = NuSVC(nu=nu, kernel=kernel)
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.svm_model = model

                    st.success(f"Nu-SVM trained with nu={nu} and kernel={kernel}")
                    for metric in metrics:
                        st.text(functions.final_evaluation(
                            st.session_state.svm_model,
                            st.session_state.X_train, st.session_state.X_test,
                            st.session_state.y_train, st.session_state.y_test,
                            metric
                        ))
        elif st.session_state.svm_mode == "grid":
            st.subheader("GridSearchCV")
            verbose = st.slider("Verbose (you will see the progress in the terminal)", 0, 12, 1, key="svm_verbose")
            param_grid = {
                'nu': np.arange(0.05, 1.0, 0.05),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
            nu_range = st.text_input("Range for nu (default: 0.05,1.0,0.05)", placeholder="example: 0.1 , 0.9 , 0.1 (from 0.1 to 0.9 with jumps of 0.1)", key="svm_nu_input").split(',')
            kernel_range = st.selectbox("Kernel", options=['linear', 'poly', 'rbf', 'sigmoid'], key="svm_kernel_input")
            def _try_float(val, default):
                try:
                    return float(val.strip())
                except (ValueError, AttributeError):
                    return default
            nu_start = _try_float(nu_range[0], 0.05) if len(nu_range) > 0 else 0.05
            nu_end   = _try_float(nu_range[1], 1.0)  if len(nu_range) > 1 else 1.0
            nu_step  = _try_float(nu_range[2], 0.05) if len(nu_range) > 2 else 0.05
            param_grid['nu'] = np.arange(nu_start, nu_end, nu_step)
            param_grid['kernel'] = kernel_range if len(kernel_range) > 0 else ['linear', 'poly', 'rbf', 'sigmoid']
            k = st.text_input("Number of folds for K-Fold (default: 4) (if you dont want CV put 0)", key="svm_kfold_input")
            n_splits = int(k) if k.isdigit() else 4
            metric = st.selectbox("Select ONE metric for evaluation during GridSearch", options=["accuracy", "precision", "recall", "f1"], key="svm_grid_metrics")
            if st.button("Run Search"):
                st.info("Running GridSearch...")
                st.session_state.svm_model, st.session_state.svm_grid = functions.svm(X_train=st.session_state.X_train, y_train=st.session_state.y_train, X_test=st.session_state.X_test, y_test=st.session_state.y_test, metric=metric, k=n_splits, params=param_grid, verbose=verbose)
                st.success("GridSearch completed!")
                st.text(f"Best nu: {st.session_state.svm_model.nu}")
                st.text(f"Best kernel: {st.session_state.svm_model.kernel}")
                for metric in metrics:
                    st.text(functions.final_evaluation(
                        st.session_state.svm_model,
                        st.session_state.X_train, st.session_state.X_test,
                        st.session_state.y_train, st.session_state.y_test,
                        metric
                    ))
            if st.button("Show GridSearchCV Results"):
                if 'svm_grid' in st.session_state:
                    grid_results = pd.DataFrame(st.session_state.svm_grid)
                    st.dataframe(grid_results, width='stretch')
                else:
                    st.warning("Please run the GridSearch first to see results.")
            if st.button("Show Validation Curve for nu"):
                if 'svm_grid' in st.session_state:
                    param_range = param_grid['nu']
                    train_scores, test_scores = validation_curve(
                        NuSVC(kernel=st.session_state.svm_model.kernel),
                        st.session_state.X_train,
                        st.session_state.y_train,
                        param_name="nu",
                        param_range=param_range,
                        cv=n_splits,
                        scoring=metric
                    )
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(param_range, train_scores_mean, label='Training score', color='blue')
                    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='orange')
                    plt.title('Validation Curve for nu')
                    plt.xlabel('Nu')
                    plt.ylabel(metric.capitalize())
                    plt.legend(loc='best')
                    st.pyplot(plt)
                else:
                    st.warning("Please run the GridSearch first to see the validation curve.")
            if st.button("Show Validation Curve for kernel"):
                if 'svm_grid' in st.session_state:
                    param_range = param_grid['kernel']
                    train_scores, test_scores = validation_curve(
                        NuSVC(nu=st.session_state.svm_model.nu),
                        st.session_state.X_train,
                        st.session_state.y_train,
                        param_name="kernel",
                        param_range=param_range,
                        cv=n_splits,
                        scoring=metric
                    )
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(param_range, train_scores_mean, label='Training score', color='blue')
                    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='orange')
                    plt.title('Validation Curve for kernel')
                    plt.xlabel('Kernel')
                    plt.ylabel(metric.capitalize())
                    plt.legend(loc='best')
                    st.pyplot(plt)
                else:
                    st.warning("Please run the GridSearch first to see the validation curve.")
    elif model_name == "Nearest Centroid":
        metrics = st.multiselect("Select metrics", options=["accuracy", "precision", "recall", "f1"], key="nc_metrics")
        if st.button("Save Model to My Account", key="nc_save"):
            user = st.session_state["username"]
            user_path = f"saved_models/{user}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if 'nc_model' in st.session_state:
                file_path = f"{user_path}/ncc_model.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(st.session_state.nc_model, f)
                st.success(f"Model saved in your folder: {file_path}")
            else:
                st.warning("Please train a model first before saving.")
        if st.button("Train Nearest Centroid Model"):
            model = NearestCentroid()
            model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.nc_model = model
            st.success("Nearest Centroid model trained!")
            for metric in metrics:
                st.text(functions.final_evaluation(
                    st.session_state.nc_model,
                    st.session_state.X_train, st.session_state.X_test,
                    st.session_state.y_train, st.session_state.y_test,
                    metric
                ))
def my_models():
    user = st.session_state["username"]
    user_path = f"saved_x_test/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as x_testf:
                st.session_state.X_test = pickle.loads(x_testf.read())
    user_path = f"saved_y_test/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as y_testf:
                st.session_state.y_test = pickle.loads(y_testf.read())
    user_path = f"saved_X_train/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as X_trainf:
                st.session_state.X_train = pickle.loads(X_trainf.read())
    user_path = f"saved_y_train/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as y_trainf:
                st.session_state.y_train = pickle.loads(y_trainf.read())


    st.title("My Models")
    user = st.session_state["username"]
    user_path = f"saved_models/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            st.subheader("Your Saved Models:")
            remove_model = st.selectbox("Select a model to remove", options=["None"] + saved_files, key="remove_model_select")
            if st.button("Remove Selected Model"):
                if remove_model != "None":
                    os.remove(f"{user_path}/{remove_model}")
                    saved_files.remove(remove_model)
                    st.success(f"Model {remove_model} has been removed.")
            for file in saved_files:
                with open(f"{user_path}/{file}", 'rb') as modelf:
                        model = pickle.loads(modelf.read())
                st.markdown("---")
                st.text(file[:-10].upper())
                if file[:-10] == "knn":
                    st.text(f"  n_neighbors: {model.get_params()['n_neighbors']}")
                elif file[:-10] == "decision_tree":
                    st.text(f"  max_depth: {model.get_params()['max_depth']} , min_samples_leaf: {model.get_params()['min_samples_leaf']}")
                elif file[:-10] == "random_forest":
                    st.text(f"  n_estimators: {model.get_params()['n_estimators']} , max_depth: {model.get_params()['max_depth']} , min_samples_leaf: {model.get_params()['min_samples_leaf']}")
                elif file[:-10] == "svm":
                    st.text(f"  nu: {model.get_params()['nu']} , kernel: {model.get_params()['kernel']}")
                elif file[:-10] == "ncc":
                    pass
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    st.text(functions.final_evaluation(model,st.session_state.X_train, st.session_state.X_test,st.session_state.y_train, st.session_state.y_test, metric))
                st.markdown("---")
        else:
            st.info("You don't have any saved models yet.")
    else:
        st.info("You don't have any saved models yet.")
def  save_data():
    st.title("Save Data")
    data  = st.multiselect("Select data to save", options=["train set", "test set" ,"full dataset"], key="save_data_select")
    save_as_name = st.text_input("save as", value="", key="save_name_select") + ".csv"
    df = st.session_state.df
    for i in data:
        if i == "train set":
            train_df = st.session_state.X_train.copy()
            train_df['target'] = st.session_state.y_train
            csv = train_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Train Set as CSV",
                data=csv,
                file_name=save_as_name,
                mime='text/csv',
                key="dl_train"
            )
        elif i == "test set":
            test_df = st.session_state.X_test.copy()
            test_df['target'] = st.session_state.y_test
            csv = test_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Test Set as CSV",
                data=csv,
                file_name=save_as_name,
                mime='text/csv',
                key="dl_test"
            )
        elif i == "full dataset":
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Dataset as CSV",
                data=csv,
                file_name=save_as_name,
                mime='text/csv',
                key="dl_full"
            )
def get_needed():
    features = None
    X_test_before_scaling = None
    encode_method = None
    scaling_method = None
    encode_columns = None
    scaled_columns = None
    user = st.session_state["username"]
    user_path = f"saved_features/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as featuresf:
                features = pickle.loads(featuresf.read())
    user_path = f"saved_X_test_before_scaling/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as X_test_before_scalingf:
                X_test_before_scaling = pickle.loads(X_test_before_scalingf.read())
    user_path = f"saved_scaling_method/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as scaling_methodf:
                scaling_method = pickle.loads(scaling_methodf.read())
    user_path = f"saved_encoded_columns/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as encode_columnsf:
                encode_columns = pickle.loads(encode_columnsf.read())
    user_path = f"saved_scaled_columns/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as scaled_columnsf:
                scaled_columns = pickle.loads(scaled_columnsf.read())
    user_path = f"saved_encode_method/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        if saved_files:
            with open(f"{user_path}/{saved_files[0]}", 'rb') as encode_methodf:
                encode_method = pickle.loads(encode_methodf.read())

    return features, X_test_before_scaling, scaling_method , encode_method , scaled_columns  , encode_columns

def predict():
    st.session_state.features, st.session_state.X_test_before_scaling, st.session_state.scaling_method , st.session_state.encode_method , st.session_state.scaled_columns  , st.session_state.encode_columns = get_needed()
    st.title("Predict")
    user = st.session_state["username"]
    user_path = f"saved_models/{user}"
    if os.path.exists(user_path):
        saved_files = os.listdir(user_path)
        selected_file = st.selectbox("Load one of your saved models:", saved_files)
        if st.button("Load Model"):
            with open(f"{user_path}/{selected_file}", 'rb') as modelf:
                model = pickle.loads(modelf.read())
            st.success(f"Successfully loaded {selected_file}!")
            st.session_state.loaded_model = model
    else:
        st.info("You don't have any saved models yet.")
    with st.form("prediction_form"):
        input_data = {}
        for feature in st.session_state.features:
            value = st.number_input(f"Enter value for {feature}:", key=f"predict_{feature}")
            input_data[feature] = [value]
        input_df = pd.DataFrame(input_data)
        input_df = pd.concat([input_df , st.session_state.X_test_before_scaling], axis =0, ignore_index=True)
        columns_to_encode = [col for col in (st.session_state.encode_columns or []) if col in input_df.columns]
        columns_to_scale = [col for col in (st.session_state.scaled_columns or []) if col in input_df.columns]
        submit = st.form_submit_button("Predict")
        if submit:
            if st.session_state.scaling_method == "Standard Scaling":
                input_df = functions.standardscaler(input_df, columns_to_scale)[0]
            elif st.session_state.scaling_method == "Min-Max Scaling":
                input_df = functions.minmaxscalar(input_df, columns_to_scale)[0]


            if st.session_state.encode_method:
                for col in columns_to_encode:
                    if st.session_state.encode_method == "Label Encoding":
                        input_df = functions.label_encoding(input_df, [col])
                    elif st.session_state.encode_method == "One-Hot Encoding":
                        input_df = functions.one_hot_encoding(input_df, [col])
            input_df = input_df.loc[0, st.session_state.features].to_frame().T
            try:
                model = st.session_state.loaded_model
                prediction = model.predict(input_df)
                st.success(f"Predicted: {prediction[0]}")
                #{st.session_state.target}
            except Exception as e:
                    st.info(f"Error during prediction: {e}")

        

# Load user data from your YAML file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# 2. Create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
authenticator.login(location='main')

# Then check the status using session_state
if st.session_state["authentication_status"]:
    st.sidebar.write(f'Welcome, *{st.session_state["name"]}*')
    authenticator.logout('Logout', 'sidebar')
    st.title("AI Model Maker")
    st.subheader("by Tomer Mashiah")
    st.markdown("---") 
    pg = st.navigation([
        st.Page(load_data, title="Load Data", icon="📂"),
        st.Page(EDA, title="EDA", icon="📊"),
        st.Page(plots, title="Plots", icon="📈"),
        st.Page(preprocessing, title="Preprocessing", icon="⚙️"),
        st.Page(split_data, title="Split Data", icon="🔄"),
        st.Page(make_model, title="Make Model", icon="🤖"),
        st.Page(my_models, title="My Models", icon="📁"),
        st.Page(predict, title="Predict", icon="🔮"),
        st.Page(save_data, title="Save Data", icon="💾"),
    ])
    pg.run()

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
    if st.checkbox("New user? Register here"):
        try:
            if authenticator.register_user(location='main'):
                st.success('User registered successfully')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(f"Registration error: {e}")