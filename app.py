import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report,silhouette_score,confusion_matrix
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


st.title("Data Science Project")

file = st.sidebar.file_uploader("Upload your csv file",type = ["csv"])
if file is not None:
    df = pd.read_csv(file)
    st.subheader("A view of your data")
    st.dataframe(df.head())
    st.subheader("Checking missing values")
    missing = df.isnull().sum()
    st.dataframe(missing)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            mode_value  = df[col].mode()[0]
            df[col].fillna(mode_value,inplace = True)
            st.success("MIssing values filled with mode")
    st.subheader("Essential information about your data wtih Data types")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str  = buffer.getvalue()
    st.text(info_str)
    st.subheader("Statistical Information about your Data ")
    st.dataframe(df.describe())

    learning_type = st.sidebar.radio("Choose Machine learning types",["Supervised","Unsupervised"])

    if learning_type == "Supervised":

        selected_target = st.sidebar.selectbox(
        label="Select the column you want to predict:",
        options=df.columns,
        placeholder="Choose one column..."
        )
        if selected_target:
            y = df[selected_target]
            x = df.drop(columns = selected_target)

            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state = 42)

        Supervised_Algorithm = st.sidebar.selectbox("Which Machine Learning Supervised_Algorithm",["Decision Tree Classifier","Random Forest Classifier","Support Vector Machine","Logistics Regression"])
        st.success(f"You Choose {Supervised_Algorithm}")
        if Supervised_Algorithm == "Decision Tree Classifier":
            random_state = st.sidebar.slider("Random state" , min_value=1, max_value=500,step = 1)
            crit = st.sidebar.segmented_control("Criterion", ["gini", "entropy", "log_loss"],default = "gini")
            max_dp1 = st.sidebar.slider("Max depth",min_value = 3,max_value = 10,step = 1)

            model = DecisionTreeClassifier(criterion = crit,random_state = random_state,max_depth = max_dp1)
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test) 
            st.subheader("Classification Report")
            plt.title("Confussion Matrix")
            sns.heatmap(confusion_matrix(y_test,y_pred))
            plt.tight_layout()
            plt.show()
            report = classification_report(y_test,y_pred)
            st.write(report)
            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")

            st.pyplot(plt)

        elif Supervised_Algorithm == "Random Forest Classifier":

            num_tree = st.sidebar.slider("Number of trees",min_value=1, max_value=500,step = 1)
            max_dp2 = st.sidebar.slider("Max depth",min_value = 3,max_value = 10,step = 1)
            random_state2 = st.sidebar.slider("Random State",min_value=1, max_value=500,step = 1)

            model = RandomForestClassifier(n_estimators = num_tree,max_depth=max_dp2,random_state=random_state2)
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test) 
            st.subheader("Classification Report")
            plt.title("Confussion Matrix")
            sns.heatmap(confusion_matrix(y_test,y_pred))
            plt.tight_layout()
            plt.show()
            report1 = classification_report(y_test,y_pred)
            st.write(report1)

        elif Supervised_Algorithm == "Suppport Vector Machine":
            model = SVC()

            model.fit(x_train,y_train)
            y_pred = model.predict(x_test) 
            st.subheader("Classification Report")
            plt.title("Confussion Matrix")
            sns.heatmap(confusion_matrix(y_test,y_pred))
            plt.tight_layout()
            plt.show()
            report2 = classification_report(y_test,y_pred)

            st.write(report2)
            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")

            st.pyplot(plt)

        elif Supervised_Algorithm == "Logistics Regression":
                
                C = st.sidebar.select_slider(
                "Select Regularization Strength (C)",
                options=[0.001, 0.01, 0.1, 1, 10, 100],
                value=1.0
                    )

                penalty = st.sidebar.selectbox(
                "Select Penalty Type",
                options=["l1", "l2"]
                )

                solver = st.sidebar.selectbox(
                "Select Solver",
                options=["liblinear", "saga"]
                    )

                max_iter = st.sidebar.slider(
                "Max Iteration2s",
                min_value=100,
                max_value=1000,
                value=200
                    )
                random_state3 = st.sidebar.slider(
                    "Random State",
                    min_value=1,
                    max_value=1000,
                    step=1

                )
                model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
                model.fit(x_train,y_train)
                y_pred = model.predict(x_test)
                
                st.subheader("Classification Report")
                report3 = classification_report(y_test,y_pred)
                st.write(report3)
                plt.figure(figsize=(5, 4))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")

                st.pyplot(plt)


    elif learning_type == "Unsupervised":
        unsup_algo = st.sidebar.radio(
        "Choose Algorithm",
        options=["KMeans Clustering", "Agglomerative Clustering", "DBSCAN Clustering"]
        ) 
        scaler = StandardScaler()

        if unsup_algo:
            choice = st.sidebar.radio("Choice",
                             options=["Cluster a column","Cluster full data"])
            if choice == "Cluster a column":
                selected_column = st.sidebar.selectbox("Chosse Which column to cluster",options = df.columns)

                data = df[[selected_column]]
                scaled_data1 = scaler.fit_transform(data)
                scaled_df = pd.DataFrame(scaled_data1, columns=[selected_column])
                if unsup_algo == "KMeans Clustering":
                    num_clusters_kmean1 = st.sidebar.slider("Number of Clusters",
                                        min_value = 2,
                                        max_value = 1000
                                        )
                            
                    iteration = st.sidebar.slider("Max Iteration2",
                                        min_value = 300,
                                        max_value = 1000
                                        )
                            
                    randomstate = st.sidebar.slider("Random State",
                                        min_value = 1,
                                        max_value = 100
                                        )
                    K_mean = KMeans(n_clusters = num_clusters_kmean1,random_state=randomstate,max_iter = iteration)
                    K_mean.fit(scaled_data1)
                    pred = K_mean.predict(scaled_data1)
                    st.subheader("Silhouette Score")
                    silhtee_score_kemans = silhouette_score(scaled_data1,pred)
                    st.write(silhtee_score_kemans)    


                    scaled_df['Cluster'] = pred

                    st.subheader("Cluster Scatter Plot")
                    fig, ax = plt.subplots()
                    sns.scatterplot(
                        data=scaled_df,
                        x=scaled_df.index,  # index as x-axis
                        y=selected_column,
                        hue="Cluster",
                        palette="viridis",
                        ax=ax
                    )
                    ax.set_xlabel("Index")
                    ax.set_ylabel(f"{selected_column} (Scaled)")
                    ax.set_title("KMeans Clustering Scatter Plot")
                    st.pyplot(fig)

                elif unsup_algo == "Agglomerative Clustering":
                    max_cluster_aggo = st.sidebar.slider("Number of Clusters",
                                        min_value = 2,
                                        max_value = 100
                                        )
                    agglo_algo = AgglomerativeClustering(n_clusters = max_cluster_aggo)
                    pred_agg = agglo_algo.fit_predict(scaled_data1)
                    st.subheader("Silhouette Score")
                    silhtee_score_agg = silhouette_score(scaled_data1,pred_agg)
                    st.write(silhtee_score_agg)
                    st.subheader("Cluster Scatter Plot")
                    fig, ax = plt.subplots()
                    sns.scatterplot(
                        x=scaled_df.index, 
                        y=scaled_df[selected_column], 
                        palette="viridis", 
                        ax=ax
                    )
                    ax.set_xlabel("Index")
                    ax.set_ylabel(selected_column)
                    ax.set_title("KMeans Clustering Scatter Plot")
                    st.pyplot(fig)
                elif unsup_algo == "DBSCAN Clustering":
                    min_clusters = st.sidebar.slider("Minimum CLusters",min_value = 2,max_value = 100,)
                    eps = st.sidebar.slider("Chosse the eps",min_value = 0.1,max_value =1.0)
                    dnscan = DBSCAN(min_samples = min_clusters,eps = eps)
                    pred_db = dnscan.fit_predict(scaled_data1)        
                    st.subheader("Silhouette Score")
                    silhtee_score_db = silhouette_score(scaled_data1,pred_db)
                    st.write(silhtee_score_db)
            elif choice == "Cluster full data":    
                scaled_data2 = scaler.fit_transform(df)
                pca = PCA(n_components = 2,random_state = 42)
                x_pca  = pca.fit_transform(scaled_data2)
                if unsup_algo == "KMeans Clustering":
                    num_clusters_kmean2 = st.sidebar.slider("Number of Clusters",
                                        min_value = 2,
                                        max_value = 1000
                                        )
                            
                    iteration2 = st.sidebar.slider("Max Iteration",
                                        min_value = 300,
                                        max_value = 1000
                                        )
                            
                    randomstate2 = st.sidebar.slider("Random State",
                                        min_value = 1,
                                        max_value = 100
                                        )
                    K_mean2 = KMeans(n_clusters = num_clusters_kmean2,random_state=randomstate2,max_iter = iteration2)
                    K_mean2.fit(x_pca)
                    pred2 = K_mean2.predict(x_pca)                  
                    st.subheader("Silhouette Score")
                    silhtee_score_kemans2 = silhouette_score(x_pca,pred2)
                    st.write(silhtee_score_kemans2)    
                elif unsup_algo == "Agglomerative Clustering":
                    max_cluster_aggo2 = st.sidebar.slider("Number of Clusters",
                                        min_value = 2,
                                        max_value = 100
                                        )
                    agglo_algo2 = AgglomerativeClustering(n_clusters = max_cluster_aggo2)
                    pred_agg2 = agglo_algo2.fit_predict(scaled_data2)
                    st.subheader("Silhouette Score")
                    silhtee_score_agg2 = silhouette_score(scaled_data2,pred_agg2)
                    st.write(silhtee_score_agg2)
                elif unsup_algo == "DBSCAN Clustering":
                    # min_clusters2 = st.sidebar.slider("Minimum CLusters",min_value = 2,max_value = 100,)
                    # eps2 = st.sidebar.slider("Chosse the eps",min_value = 0.1,max_value =1.0)
                    dnscan2 = DBSCAN(min_samples = 2,eps = 0.5)
                    pred_db2 = dnscan2.fit_predict(scaled_data2)      
                    st.subheader("Silhouette Score")
                    silhtee_score_db2 = silhouette_score(scaled_data2,pred_db2)
                    st.write(silhtee_score_db2)


            # use_scaler = st.toggle("Cluster the full dataset")
            