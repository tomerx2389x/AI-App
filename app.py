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
import streamlit as st
import pandas as pd

import functions as f
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
            except Exception as e:
                st.warning(f"Error loading CSV: {e}")
            st.session_state.outliners_dict = {"column": [],"count": []}
            for i in st.session_state.df.select_dtypes(include=np.number).columns:
                text , count = f.outliners(st.session_state.df,i)
                if count > 0:
                    st.session_state.outliners_dict["column"].append(i)
                    st.session_state.outliners_dict["count"].append(count)

            st.dataframe(st.session_state.df,
            width='stretch')
def EDA():
    if "df" not in st.session_state:
        st.warning("Please load data first!")
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
            st.write(f.null(df))
        elif func == "outliers":
            st.session_state.outliners_dict = {"column": [],"count": []}
            for i in df.select_dtypes(include=np.number).columns:
                text , count = f.outliners(df,i)
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
            st.write(f.duplicates(df))
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
        df = st.session_state.df
        st.session_state.encode_method = None
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
                        st.session_state.df = f.fill_uniform(st.session_state.df, col_to_fix)
                    
                    st.rerun()

        elif func == "outliers":
            outliners_dict = st.session_state.outliners_dict
            column = st.selectbox("Select column to remove outliers", options=list(outliners_dict["column"]), key="outlier_col")
            st.write(f"Number of outliers in {column}: {outliners_dict['count'][outliners_dict['column'].index(column)]}")
            if st.button("Remove Outliers"):
                # Update session state directly
                st.session_state.df = f.removeoutliners(st.session_state.df, column)
                st.rerun()
                
            if st.button("squash outliers"):
                st.session_state.df = f.squashoutliners(st.session_state.df, column)
                st.rerun()
    
        elif func == "duplicates":
            st.write(f.duplicates(df))
            if st.button("Remove Duplicates"):
                st.session_state.df = f.removeduplicates(df)
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

            # 3. The Logic Trigger
            if st.button("Encode"):
                # Perform the transformation
                if method == "Label Encoding":
                    st.session_state.df, st.session_state.encoder = f.lable_encoding(st.session_state.df, column)
                else:
                    st.session_state.df, st.session_state.encoder = f.one_hot_encoding(st.session_state.df, [column])
                
                # Notify the user
                st.success(f"Encoded {column}!")
                # No need for st.rerun() here; the button click already handles the refresh!
        elif func == "scaling":
            column = st.multiselect("Select column to scale", options=df.select_dtypes(include=np.number).columns, key="scale_col")
            method = st.selectbox("Select scaling method", options=["Standard Scaling", "Min-Max Scaling"], key="scale_method")
            st.session_state.scaling_method = None
            if st.button("Scale"):
                st.session_state.DF_BEFORE_SCALING = df.copy()
                if method == "Standard Scaling":
                    st.session_state.df , st.session_state.scaler = f.standardscaler(df, column)
                elif method == "Min-Max Scaling":
                    st.session_state.df   , st.session_state.scaler = f.minmaxscalar(df, column)
                st.rerun()
            st.session_state.scaling_method = method
        
        elif func == "feature_engineering":
            st.write("Feature engineering functionality coming soon!")
        st.session_state.df =  st.session_state.df.copy()
        st.dataframe(st.session_state.df, width='stretch')
        st.write(f"Total Rows: {st.session_state.df.shape[0]} | Total Columns: {st.session_state.df.shape[1]}")
def split_data():
    st.title("Split Data")
    df = st.session_state.df
    
    y_column = st.selectbox("Select target column", options=df.columns, key="model_target_col")
    x_columns = st.multiselect("Select feature columns", options=[col for col in df.columns if col != y_column], key="model_feature_cols")
    st.session_state.target = y_column
    st.session_state.features = x_columns
    test_size = st.slider("Select test size", 0.1, 0.9, 0.2, key="train_size_slider")

    # 1. Split data ONLY if it hasn't been done or if parameters change
    if st.button("Initialize / Reset Split"):
        y = df[y_column]
        X = df[x_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.session_state.X_test_before_scaling = st.session_state.DF_BEFORE_SCALING[x_columns].loc[X_test.index]
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.success("Data split successfully!")

    # Ensure data exists before trying to sample or plot
    if "X_train" in st.session_state:
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Apply SMOTE"):
                st.session_state.X_train, st.session_state.y_train = f.oversampling(st.session_state.X_train, st.session_state.y_train)
                st.success("Balanced with SMOTE")

        with col2:
            if st.button("Apply Undersampling"):
                st.session_state.X_train, st.session_state.y_train = f.undersampling(st.session_state.X_train, st.session_state.y_train)
                st.success("Balanced with Undersampling")

        # 2. The Plotting Logic
        st.divider()
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(x=st.session_state.y_train, ax=ax)
        st.pyplot(fig)


def make_model():
    st.title("Make Model")
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
                        st.text(f.final_evaluation(
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
                st.session_state.knn_model , st.session_state.knn_grid = f.knn(X_train=st.session_state.X_train, y_train=st.session_state.y_train,X_test = st.session_state.X_test,y_test = st.session_state.y_test , metric = metric , k= n_splits , params=param_grid , verbose=verbose)
                st.success("GridSearch completed!")
                st.text(f"Best k: {st.session_state.knn_model.n_neighbors}")
                for metric in metrics:
                    st.text(f.final_evaluation(
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
                        st.text(f.final_evaluation(
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
                st.session_state.dt_model , st.session_state.dt_grid = f.tree(X_train=st.session_state.X_train, y_train=st.session_state.y_train,X_test = st.session_state.X_test,y_test = st.session_state.y_test , metric = metric , k= n_splits , params=param_grid , verbose=verbose)
                st.success("GridSearch completed!")
                st.text(f"Best max_depth: {st.session_state.dt_model.max_depth}")
                st.text(f"Best min_samples_leaf: {st.session_state.dt_model.min_samples_leaf}")
                for metric in metrics:
                    st.text(f.final_evaluation(
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
        metrics = st.multiselect("Select metrics", options=["accuracy", "precision", "recall", "f1"], key="dt_metrics")
        if 'rf_mode' not in st.session_state:
            st.session_state.rf_mode = None
        col1, col2 = st.columns(2)
        if col1.button("Manual Entry"):
            st.session_state.rf_mode = "manual"
        if col2.button("Grid Search"):
            st.session_state.rf_mode = "grid"
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
                        st.text(f.final_evaluation(
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
                st.session_state.rf_model , st.session_state.rf_grid = f.random_forest(X_train=st.session_state.X_train, y_train=st.session_state.y_train,X_test = st.session_state.X_test,y_test = st.session_state.y_test , metric = metric , k= n_splits , params1=params1, params2=params2 , verbose=verbose)
                st.success("GridSearch completed!")
                st.text(f"Best n_estimators: {st.session_state.rf_model.n_estimators}")
                st.text(f"Best max_depth: {st.session_state.rf_model.max_depth}")
                st.text(f"Best min_samples_leaf: {st.session_state.rf_model.min_samples_leaf}")
                for metric in metrics:
                    st.text(f.final_evaluation(
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
        if st.session_state.svm_mode == "manual":
            st.subheader("Manual")
            with st.form("svm_manual_form"):
                C = st.number_input("Regularization parameter", min_value=0.01, value=1.0, step=0.1)
                kernel = st.selectbox("Kernel", options=["linear", "poly", "rbf", "sigmoid"], key="svm_kernel")
                submit_manual = st.form_submit_button("Train SVM Model")

                if submit_manual:
                    model = SVC(C=C, kernel=kernel)
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.svm_model = model

                    st.success(f"SVM trained with C={C} and kernel={kernel}")
                    for metric in metrics:
                        st.text(f.final_evaluation(
                            st.session_state.svm_model,
                            st.session_state.X_train, st.session_state.X_test,
                            st.session_state.y_train, st.session_state.y_test,
                            metric
                        ))
        elif st.session_state.svm_mode == "grid":
            st.subheader("GridSearchCV")
            verbose = st.slider("Verbose (you will see the progress in the terminal)", 0, 12, 1, key="svm_verbose")
            param_grid = {
                'C':np.arange(0.01, 1.01, 0.1),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
            C_range = st.text_input("Range for C (default: 0.01,1.01,0.1)", placeholder="example:0.01 , 1.01 , 0.1 (from 0.01 to 1.01 with jumps of 0.1)", key="svm_C_input").split(',')
            kernel_range = st.selectbox("Range for kernel (default: linear,poly,rbf,sigmoid)", options=['linear', 'poly', 'rbf', 'sigmoid'], key="svm_kernel_input")
            param_grid['C'] = np.arange(float(C_range[0]), float(C_range[1]), float(C_range[2]))
            param_grid['kernel'] = kernel_range if len(kernel_range) > 0 else ['linear', 'poly', 'rbf', 'sigmoid']
            C_start = int(C_range[0]) if len(C_range) > 0 and C_range[0].strip().isdigit() else 1
            C_end = int(C_range[1]) + 1 if len(C_range) > 1 and C_range[1].strip().isdigit() else 21
            step = int(C_range[2]) if len(C_range) > 2 and C_range[2].strip().isdigit() else 1
            param_grid['C'] = np.arange(C_start, C_end, step)
            k = st.text_input("Number of folds for K-Fold (default: 4) (if you dont want CV put 0)", key="dt_kfold_input")
            n_splits = int(k) if k.isdigit() else 4
            metric = st.selectbox("Select ONE metric for evaluation during GridSearch", options=["accuracy", "precision", "recall", "f1"], key="dt_grid_metrics")
            if st.button("Run Search"):
                st.info("Running GridSearch...")
                st.session_state.svm_model , st.session_state.svm_grid = f.svm(X_train=st.session_state.X_train, y_train=st.session_state.y_train,X_test = st.session_state.X_test,y_test = st.session_state.y_test , metric = metric , k= n_splits , params=param_grid , verbose=verbose)
                st.success("GridSearch completed!")
                st.text(f"Best C: {st.session_state.svm_model.C}")
                st.text(f"Best kernel: {st.session_state.svm_model.kernel}")
                for metric in metrics:
                    st.text(f.final_evaluation(
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
            if st.button("Show Validation Curve for C"):
                if 'svm_grid' in st.session_state:
                    param_range = param_grid['C']
                    train_scores, test_scores = validation_curve(
                        SVC(kernel=st.session_state.svm_model.kernel),
                        st.session_state.X_train,
                        st.session_state.y_train,
                        param_name="C",
                        param_range=param_range,
                        cv=n_splits,
                        scoring=metric
                    )
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(param_range, train_scores_mean, label='Training score', color='blue')
                    plt.plot(param_range, test_scores_mean, label='Cross-validation score', color='orange')
                    plt.title('Validation Curve for C')
                    plt.xlabel('Regularization Parameter (C)')
                    plt.ylabel(metric.capitalize())
                    plt.legend(loc='best')
                    st.pyplot(plt)
                else:
                    st.warning("Please run the GridSearch first to see the validation curve.")
            if st.button("Show Validation Curve for kernel"):
                if 'svm_grid' in st.session_state:
                    param_range = param_grid['kernel']
                    train_scores, test_scores = validation_curve(
                        SVC(C=st.session_state.svm_model.C),
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
        if st.button("Train Nearest Centroid Model"):
            model = NearestCentroid()
            model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.nc_model = model
            st.success("Nearest Centroid model trained!")
            for metric in metrics:
                st.text(f.final_evaluation(
                    st.session_state.nc_model,
                    st.session_state.X_train, st.session_state.X_test,
                    st.session_state.y_train, st.session_state.y_test,
                    metric
                ))
def  save_data():
    st.title("Save Data")
    data  = st.multiselect("Select data to save", options=["train set", "test set" ,"full dataset"], key="save_data_select")
    save_as_name = st.text_input("save as", value="", key="save_name_select") + ".csv"
    df = st.session_state.df
    for i in data:
        if i == "train set":
            train_df = st.session_state.X_train.copy()
            train_df['target'] = st.session_state.y_train
            if st.button("Download Train Set as CSV"):
                csv = train_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Train Set as CSV",
                    data=csv,
                    file_name=save_as_name,
                    mime='text/csv',
                )
                st.success("Your train data has been and downloaded successfully!")   
        elif i == "test set":
            test_df = st.session_state.X_test.copy()
            test_df['target'] = st.session_state.y_test 
            if st.button("Download Test Set as CSV"):
                csv = test_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Test Set as CSV",
                    data=csv,
                    file_name=save_as_name,
                    mime='text/csv',
                )
                st.success("Your test data has been and downloaded successfully!")
        elif i == "full dataset":
            if st.button("Download Full Dataset as CSV"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Processed Data as CSV",
                    data=csv,
                    file_name=save_as_name,
                    mime='text/csv',
                )
                st.success("Your data has been and downloaded successfully!")
def predict():
    st.title("Predict")
    model_name = st.selectbox("Select model for prediction", options=["KNN", "Decision Tree", "Random Forest", "SVM", "Nearest Centroid"], key="predict_model_select")
    if model_name == "KNN":
        if 'knn_model' in st.session_state:
            model = st.session_state.knn_model
        else:
            st.warning("Please train a KNN model first on the 'Make Model' page.")
    elif model_name == "Decision Tree":
        if 'dt_model' in st.session_state:
            model = st.session_state.dt_model
        else:
            st.warning("Please train a Decision Tree model first on the 'Make Model' page.")
    elif model_name == "Random Forest":
        if 'rf_model' in st.session_state:
            model = st.session_state.rf_model
        else:
            st.warning("Please train a Random Forest model first on the 'Make Model' page.")
    elif model_name == "SVM":
        if 'svm_model' in st.session_state:
            model = st.session_state.svm_model
        else:
            st.warning("Please train a SVM model first on the 'Make Model' page.")
    elif model_name == "Nearest Centroid":
        if 'nc_model' in st.session_state:
            model = st.session_state.nc_model
        else:
            st.warning("Please train a Nearest Centroid model first on the 'Make Model' page.")
    st.write("Model loaded. You can now input values for prediction.")
    with st.form("prediction_form"):
        input_data = {}
        for feature in st.session_state.features:
            value = st.number_input(f"Enter value for {feature}:", key=f"predict_{feature}")
            input_data[feature] = [value]
        input_df = pd.DataFrame(input_data)
        input_df = pd.concat([input_df , st.session_state.X_test_before_scaling], axis =0, ignore_index=True)
        submit = st.form_submit_button("Predict")

        if submit:
            if st.session_state.scaling_method == "Standard Scaling":
                input_df = f.standardscaler(input_df, st.session_state.features)[0]
            elif st.session_state.scaling_method == "Min-Max Scaling":
                input_df = f.minmaxscalar(input_df, st.session_state.features)[0]


            if st.session_state.encode_method:
                for col in input_df.columns:
                    if st.session_state.encode_method == "Label Encoding":
                        input_df = f.lable_encoding(input_df, [col])
                    elif st.session_state.encode_method == "One-Hot Encoding":
                        input_df = f.one_hot_encoding(input_df, [col])
            input_df = input_df.loc[0, st.session_state.features].to_frame().T
            try:
                prediction = model.predict(input_df)
                st.success(f"Predicted {st.session_state.target}: {prediction[0]}")
                st.dataframe(input_df)
            except Exception as e:
                    st.error(f"Error during prediction: {e}")

        

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
    st.Page(predict, title="Predict", icon="🔮"),
    st.Page(save_data, title="Save Data", icon="💾"),
   
])
pg.run()