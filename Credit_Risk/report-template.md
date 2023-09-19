# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of this analysis was to train and observe two machine learning models whith our dataset which contained around 77500 data points for our models to train on.


* Explain what financial information the data was on, and what you needed to predict.
Our data included the amount of the loan given, It's interest_rate, income and debt of the borrower, debt to income reatio, number of accounts they had and their derogatory_marks. Our machine has to be able to predict if our loan status is healthy (0) or unhealthy (1).


* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
Our data set has one flaw and that is it's bias towards the number of healthy data points. The ratio of available data points on healthy to unhealthy is 30:1. which in this case does have an effect on the models precison in determining the unhealhty loans. 


* Describe the stages of the machine learning process you went through as part of this analysis.
First we split our data into features and labels and created test and training data sets/
Then we trained our first ligsticRegression model with the imbalanced data set, which resulted in 95% accuracy in correctly predicting our data points. 
Then we used theimblearn library to oversample our data points to see if it has any difference.


* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
After a little experementation, I used the 0.18 ratio to resample the data as it seemed reasonable enough in accordance ot real wolrd and it gave me a flat 99% accuracy. 


how ever though to formulation behind oversampling, the increased accuracy of the data is thought to the data samples being the same copy. specialy since in our original data, we only had 619 unhealthy data points but with this ratio, we 10128 which menas all these new data points are copies of those 619 orogonal points. 


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
In our first model, the accuracy of our data is 95%, with a high accuracy and precision  on our healthy data data points. How ever, the these numbers are a lot lower for our unhealthy data. which reults in a low 88% f1-score. And that means the models is unable to identify some of our unhealhty loans. We try to fix this by oversampling our training data and see if it makes any difference. 


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  After oversampling the training data, Our second model still performs the same with the only difference being the recall score for our unhealthy data which goes up to 99%. This is though to the fact that our negative leans data points have been increased drastically which affects this number. how ever, the precision score is even lower and we can see that the number of our False negatives has even increased. This is though to the data points being copied and is not only overfitting the model, but resulting in a false 4$ increase inour accuracy. 

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
Between the two models, I still think our first model is more accurate in being generalized and our second model is overfitted. even though it has a high accuracy.


* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
In this case, it is important to predint both of these loans, as our precision to correctly guress the  healthy loans, results in better predicrion of creditworthyness of future borrowers. And the precision in identifying unhealthy loans results in less loss and they both have a direct role to play in minimzing the default loss of the company. So in conclusion, our it is important to look at our f-1score score. 

If you do not recommend any of the models, please justify your reasoning.
In this report I can conclude that our models are not satisfactory in meeting with the correct requirments. both our mdoels do well with predicting healthy loans. But when it comes to accurately predicting our unhealthy data, their procision falls to an average of 85%. which then affects our f1-score. My recommendations for further investing into these machin learning models is to try using a differnet approach like using random forest or if possible, add extra columns(features) to our data set which might help to imporve our models accuracy in deperating these unhealthy data points and increast our precision for unhealthy loans. And lastly, the most reliable solution to increase our models accurace is to add more unhealhty samples to our data sets to further increase the precision and there for the overall accuracy of our model.

