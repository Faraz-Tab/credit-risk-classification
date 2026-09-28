# Credit Risk Classification Report

> This report describes the original analysis (unscaled features, unstratified split). The current, reproducible results are in the [README](../README.md).

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of this analysis was to train and observe two machine learning models with our dataset, which contained around 77,500 data points for our models to train on.


* Explain what financial information the data was on, and what you needed to predict.
Our data included the amount of the loan given, its interest rate, the income and debt of the borrower, the debt-to-income ratio, the number of accounts they had, and their derogatory marks. Our model has to be able to predict whether a loan is healthy (0) or unhealthy (1).


* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
Our dataset has one flaw, and that is its bias towards the number of healthy data points. The ratio of healthy to unhealthy data points is 30:1, which in this case does have an effect on the model's precision in determining the unhealthy loans.


* Describe the stages of the machine learning process you went through as part of this analysis.
First, we split our data into features and labels and created test and training datasets.
Then we trained our first LogisticRegression model with the imbalanced dataset, which resulted in 95% accuracy in correctly predicting our data points.
Then we used the imblearn library to oversample our data points to see if it made any difference.


* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
After a little experimentation, I used the 0.18 ratio to resample the data, as it seemed reasonable enough in accordance with the real world, and it gave me a flat 99% accuracy.


However, due to the formulation behind oversampling, the increased accuracy is thought to be due to the data samples being copies. Especially since in our original data we only had 619 unhealthy data points, but with this ratio we have 10,128, which means all these new data points are copies of those 619 original points.


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
In our first model, the accuracy is 95%, with high accuracy and precision on our healthy data points. However, these numbers are a lot lower for our unhealthy data, which results in a low 88% F1-score. That means the model is unable to identify some of our unhealthy loans. We try to fix this by oversampling our training data to see if it makes any difference.


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  After oversampling the training data, our second model still performs the same, with the only difference being the recall score for our unhealthy data, which goes up to 99%. This is thought to be due to the fact that our unhealthy loan data points have been increased drastically, which affects this number. However, the precision score is even lower, and we can see that the number of our false negatives has even increased. This is thought to be due to the data points being copied, which is not only overfitting the model but resulting in a false 4% increase in our accuracy.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
Between the two models, I still think our first model is better at generalizing and our second model is overfitted, even though it has a high accuracy.


* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
In this case, it is important to predict both of these loans, as our precision in correctly guessing the healthy loans results in better prediction of the creditworthiness of future borrowers. And the precision in identifying unhealthy loans results in less loss, and they both have a direct role to play in minimizing the default loss of the company. So in conclusion, it is important to look at our F1-score.

If you do not recommend any of the models, please justify your reasoning.
In this report, I can conclude that our models are not satisfactory in meeting the correct requirements. Both our models do well with predicting healthy loans. But when it comes to accurately predicting our unhealthy data, their precision falls to an average of 85%, which then affects our F1-score. My recommendations for further investing in these machine learning models are to try a different approach, like using random forest, or, if possible, add extra columns (features) to our dataset, which might help to improve our model's accuracy in separating these unhealthy data points and increase our precision for unhealthy loans. And lastly, the most reliable solution to increase our model's accuracy is to add more unhealthy samples to our datasets to further increase the precision and therefore the overall accuracy of our model.
