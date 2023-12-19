# Feedback and Iteration

## After Data collect and EDA:
Which relationships would be most interesting to you regarding variables for predicting a college average student income? 
Also, which universities would be interesting for you if I were to do a test case using shape to explain why certain universities have certain fitted values? 
My dataset includes avg faculty salary, enrollment size, attendance cost, average SAT scores, completion, and retention rates, and likewise. Any thoughts off the bat?
-	“I’ve been at two different universities with very different retention rates, and I’d be interested to see if that could be explained.”
-	“This is a cool data set! It might be interesting to see if you could create a predictor of earnings after 6 years of enrollment using sex, school, cost of attendance, and faculty salary to see if you can predict how much a certain student will earn?”
-	"My initial thoughts are regarding things that the university can either directly or indirectly control
it gives a clear reason for the analysis, perhaps the university wants to claim higher post grad salaries.
From the examples you have, I would recommend avg faculty salary, attendance cost, and retention rates. Does higher faculty salary actually lead to better professors, raising the quality of the student's understanding?
Is the attendance cost worth it to students, or is it very much a case of diminishing returns? Should universities focus on keeping retention high, or does that possibly "water down" the graduates, making them overall less prepared and thus taking lower paying jobs.
To be clear, is this the average income of the college STUDENTS, or the ALUMNI?"
**This helped me choose expected income as my response variable, include dependence scatterplots in my streamlit app (which answer these questions)**

What feedback do you have on the SHAP plots I have made?
-	"How are shap plots made? How are they additive?": They take predictions and compare them when all held constant, but one input changed. **Implemented explanation into streamlit app**

Feedback from professor at private university in WA (What other aspects should I explore? What do you think of this? i.e., does this look correct and interesting?):
-	Could look into net attendance cost (after aid from university): **Found in Data dictionary, and implemented into API call**
-	Will also want to keep in mind: income measures are self-reported, so this could create some bias, unreliability in the data
-	Interested to know also: factors contributing  retention rates; would be great for a future analysis with that as the response variable

Feedback on app (Is there anything that is unclear? What feedback do you have after using this app?) : 
- What is a shap value?
- Colors of shap value seem counter intuitive, can that be fixed? : Will look into for future developments
- Some good notes on a typ0 to fix and a formatting error: **Implemented**
- Ordering list of school alphabetically would be easier to sort through: **Implemented**
- Describe which universities are included in the analysis (4-year degree awarding universities): **Implemented**
