import streamlit as st
import shap
from shap import TreeExplainer
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# SOURCE CODE
# cache data
# @st.cache_data
def data_import():
    # import data
    X_filled = pd.read_csv("X_filled.csv",index_col="School Name")
    Xtrain_filled = pd.read_csv("Xtrain_filled.csv", index_col="School Name")
    Xtest_filled = pd.read_csv("Xtest_filled.csv", index_col="School Name")
    ytrain = pd.read_csv("ytrain.csv", index_col="School Name").squeeze()
    ytest0 = pd.read_csv("ytest.csv", index_col="School Name").squeeze()
    ytrain10 = pd.read_csv("ytrain10.csv", index_col="School Name").squeeze()
    ytest010 = pd.read_csv("ytest10.csv", index_col="School Name").squeeze()
    # eventually, will call DataCollectClean script, which will produced these csvs,
    # and then will read them in as done here
    return X_filled,Xtrain_filled,ytrain

X_filled,Xtrain_filled,ytrain = data_import()

# train chosen model(s)
rf = RandomForestRegressor(n_estimators=150, criterion='squared_error').fit(Xtrain_filled,ytrain)
# rf10 = RandomForestRegressor(n_estimators=150, criterion='squared_error').fit(Xtrain_filled,ytrain10)
# NOTE: 10 year measures still have a few missing values. FIXME

# get shap values
explainer = TreeExplainer(rf)
shap_values = explainer(X_filled) # get shap values for all colleges

# APP
st.title("Interpreting College ROI")

tab1, tab2, tab3 = st.tabs(["School Specific", "Feature Importance", "Global Summary"])

with tab1:
   st.header("A cat")

with tab2:
    # feature importance plot
    st.header('Feature Importance')
    st.text("This plot helps us see the average contribution of each feature to expected income")
    # if get current figure does not work, try this:
    fig, ax = plt.subplots(nrows=1,ncols=1)
    shap.plots.bar(shap_values,max_display = 15,show=True)
    # shap.summary_plot(shap_values,plot_type = 'bar',show=False)
    # as stated in documentation, setting show to false allows for further customization
    # https://shap.readthedocs.io/en/latest/generated/shap.plots.bar.html
    #st.pyplot(plt.gcf())
    plt.xlabel("Average absolute impact on model output\n(mean(|SHAP value|))")
    # get current figure and attempt to set xlabel
    st.pyplot(fig)
    
    # FIXME: change model output to 6 or 10 year income

with tab3:
   st.header("An owl")