# College Scorecard Analysis

This repository contains the following files related to data collected from the College Scorecard government API using Python's `requests` package:

* `CollegeScorecard-CollectClean.ipynb`, containing full code to read in the apikey, create a dictionary of desired fields to pull from the api, the requests api call, and page by page for loop to create df. Baseline reference code for performing this read in is based on/used as a reference code by Clarence Li in his College-Scorecard-Data-Analysis repo ([https://github.com/kiseki1107](https://github.com/kiseki1107/College-Scorecard-Data-Analysis))
* The full resulting dataset: `college_df.csv`
* `CollegeScorecardEDA.ipynb`, containing exploratory data analysis, including graphics included in my blog post, [Analysis of College Scoreboard Data](https://runstats21.github.io/stat-386-projects/2022/11/18/csb-eda.html)

Full details on the data collection process can be found in another one of my blog posts: [College Scorecard: Accessing Higher Education Statistics with US Department of Education API](https://runstats21.github.io/stat-386-projects/2022/10/17/webscraping-post.html)

## Statistical/ML Modeling
The following files contain work used to fit both basic and advanced statistical models to effectively estimate expected income of college students (6 years post-entry):
- `simpleSupervisedOLS+Lasso.ipynb`
- `advancedSupervisedLearn.ipynb`

## SHAP: Interpretable ML/Explainable AI
`simpleModel+shap.ipynb` shows some intriguing visualizations not only in regards to factors related to expected income 6 years post college entry, but also into the ability of using the [shap](https://shap.readthedocs.io/en/latest/index.html) package to "open the black box" of ML models.

### Interactive Dashboard
In order to allow easy accessibility to see both explanations of expected incomes for thousands of US universities and understand average contributions of different variables to that income, I built a dashboard using the [streamlit](https://streamlit.io/) package. This dashboard can be found here: https://collegeroi.streamlit.app/

## Next steps:
- cluster analysis and possible implementation of clusters as new features
- including interactions and cohort groupings on shap plots in streamlit dashboard
