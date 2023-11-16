# College Scorecard Analysis

This repository contains the following files related to data collected from the College Scorecard government API using Python's `requests` package:

* `CollegeScorecard-CollectClean.ipynb`, containing full code to read in the apikey, create a dictionary of desired fields to pull from the api, the requests api call, and page by page for loop to create df
* The full resulting dataset: `college_df.csv`
* `CollegeScorecardEDA.ipynb`, containing exploratory data analysis, including graphics included in my blog post, [Analysis of College Scoreboard Data](https://runstats21.github.io/stat-386-projects/2022/11/18/csb-eda.html)

Full details on the data collection process can be found in my blog post: [College Scorecard: Accessing Higher Education Statistics with US Department of Education API](https://runstats21.github.io/stat-386-projects/2022/10/17/webscraping-post.html)

## SHAP: Interpretable ML/Explainable AI
`simpleModel+shap.ipynb` shows some intriguing visualizations not only in regards to factors related to expected income 6 years post college entry, but also into the ability of using the [shap](https://shap.readthedocs.io/en/latest/index.html) python package to "open the black box" of ML models.


