import streamlit as st
import shap
from shap import TreeExplainer
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# APP
st.title("Interpreting College ROI")

st.markdown("This app is intended to help explain both expected income and how this is estimated through machine learning.\nExplore the tabs below to find both the **expected post-entry income** of your favorite school AND *how* different variables contributed to that expected income score.")
st.markdown('Much of this work is done using the [`shap`](https://shap.readthedocs.io/en/latest/index.html) python package, which calculates the average contribution of model inputs to help open the "black-box" of Machine Learning models.')
# TODO: clean 10 year data, and edit data import accordingly
# outcome = st.radio(
#     "Select years post entry:",
#     [6, 10]
# )

# SOURCE CODE
# cache data
@st.experimental_memo
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
    
    #ytrain_chosen = ytrain if outcome == 6 else ytrain10

    return X_filled,Xtrain_filled,ytrain

X_filled,Xtrain_filled,ytrain = data_import()


# train chosen model(s)
@st.experimental_singleton
def model_fit(y):
    rf = RandomForestRegressor(n_estimators=150, criterion='squared_error').fit(Xtrain_filled,y)
    # rf10 = RandomForestRegressor(n_estimators=150, criterion='squared_error').fit(Xtrain_filled,ytrain10)
    # NOTE: 10 year measures still have a few missing values. FIXME
    return rf

rf = model_fit(ytrain)

# if time: could also fit quantile regressor! and get ci's of expected income for each school

# get shap values
# NEED to cache these shap values
@st.experimental_memo
def get_shap_values(_fitted_model, feature_set):
    explainer = TreeExplainer(_fitted_model)
    shap_values = explainer(feature_set) # get shap values for all colleges
    # example below:
    # explainer = TreeExplainer(rf)
    # shap_values = explainer(X_filled) # get shap values for all colleges
    return shap_values

shap_values = get_shap_values(_fitted_model=rf,feature_set=X_filled)

# APP continued

# if desire 2 columns
# col1, col2 = st.columns(2)


tab1, tab2, tab3 = st.tabs(["School Specific", "Feature Contributions", "Global Summary"])

# trying to edit plot dims in side bar didn't work,
# but is good template if desiring to use a side bar
# plot_width = st.sidebar.slider("plot width", 1, 25, 3)
# plot_height = st.sidebar.slider("plot height", 1, 25, 1)

with tab1:
    st.header("Local Explainations by Selected School")
    school = st.selectbox("Select a school", options = X_filled.index,
                        label_visibility="visible")
    idx_of_interest = np.argwhere(X_filled.index == f'{school}')[0][0]
    school_fig, ax1 = plt.subplots(1,1)
    shap.plots.waterfall(shap_values[idx_of_interest],show=False)
    plt.title(f'{school} Expected Income 6-years Post Graduation: Explained')
    # plt.show()
    st.pyplot(school_fig)



with tab2:
    # TODO: change y-axis label
    st.header("Feature Contribution Summary")
    disp_feat = st.selectbox('Select a feature to display', options=X_filled.columns)
    st.subheader(f'Scatterplot of {disp_feat} Contribution')
    shap.plots.scatter(shap_values[:, f'{disp_feat}'],show=True,
                       ylabel = "Expected Income contribution")
    st.pyplot(plt.gcf())

    # if time: add interaction plot
    

with tab3:
    # feature importance plot
    st.subheader('Feature Importance')
    # outcome = st.selectbox("Choose outcome of interest", options = [])
    st.markdown("This plot helps us see the average contribution of each feature to expected income")
    # if get current figure does not work, try this:
    fig, ax = plt.subplots(nrows=1,ncols=1)
    #shap.plots.bar(shap_values,max_display = 15)
    shap.summary_plot(shap_values,max_display = 15,plot_type = 'bar')
    # as stated in documentation, setting show to false allows for further customization
    # https://shap.readthedocs.io/en/latest/generated/shap.plots.bar.html
    #st.pyplot(plt.gcf())
    # get current figure and attempt to set xlabel
    plt.xlabel("Average absolute impact on model output\n(mean(|SHAP value|))")
    st.pyplot(fig) # may want to add clear_figure = True
    # FIXME: change model output to 6 or 10 year income


st.write("""
         Copyright &copy; 2023 Tyler Ward. All rights reserved.
         """)