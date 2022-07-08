# imbalanced-ml

This repo looks into the effects of changing the balance of the training data classes on the final predicted scores from the model.
The output will be a dashboard visualising how the model performance changes with train class balance, for the purpose of understanding
the effects of imbalanced data on model performance.

## Project Outline

Using the Titanic dataset from [Kaggle](https://www.kaggle.com/competitions/titanic/data) a baseline model with balanced training
data classes is created.
The class balance is changed by including more negative samples, and then removing positive samples once all negative ones have been
included in the model training.