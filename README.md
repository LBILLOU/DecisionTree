# Handmade Decision Tree :herb:

## Project Details
This short python script enables to quickly generate and visualize a decision tree generated from a training data set. Using dataframes from pandas, the scripts creates new branches by maximizing information gain from gini impurity calculations. Works well with small data sets with limited categorical values.

## Execution
    git clone
    pip3 install -r requirements.txt
    python3 DecisionTree.py

## Example
    |    ####   #####  #####  #  #####  #  #####  #    #     #####  ####   #####  #####
    |    #   #  #      #      #  #      #  #   #  ##   #       #    #   #  #      #
    |    #   #  ###    #      #  #####  #  #   #  # #  #       #    ###    ###    ###
    |    #   #  #      #      #      #  #  #   #  #  # #       #    #  #   #      #
    |    ####   #####  #####  #  #####  #  #####  #   ##       #    #   #  #####  #####
    |
    |    >>> Import your csv file (filename.csv) :
    |    iris.csv
    |
    |    > Imported Data Head
    |       sepal_length  sepal_width  petal_length  petal_width species
    |    0           5.1          3.5           1.4          0.2  setosa
    |    1           4.9          3.0           1.4          0.2  setosa
    |    2           4.7          3.2           1.3          0.2  setosa
    |    3           4.6          3.1           1.5          0.2  setosa
    |    4           5.0          3.6           1.4          0.2  setosa
    |    > Imported Data Shape    (150, 5)
    |    > Imported Data Columns  ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'    |]
    |
    |    >>> Which column from the above would you like to predict?
    |    species
    |
    |    Decision Tree creation in progress...
    |    Time to create Decision Tree : 0:00:48.427337
    |    Decision tree created from 'iris.csv'
    |    Target variable is 'species'
    |
    |
    |    #   #  #####  #    #  ####    #####  ####   #####  #####
    |     # #   #   #  #    #  #   #     #    #   #  #      #
    |      #    #   #  #    #  ###       #    ###    ###    ###
    |      #    #   #  #    #  #  #      #    #  #   #      #
    |      #    #####  ######  #   #     #    #   #  #####  #####
    |
    |    #-> Is petal_width >= 1.0?
    |    #-> YES
    |    #   #-> Is petal_width >= 1.8?
    |    #   #-> YES
    |    #   #   #-> Is petal_length >= 4.9?
    |    #   #   #-> YES
    |    #   #       Prediction(s) - {'virginica': '100 %'}
    |    #   #   #-> NO
    |    #   #       #-> Is sepal_width >= 3.2?
    |    #   #       #-> YES
    |    #   #           Prediction(s) - {'versicolor': '100 %'}
    |    #   #       #-> NO
    |    #   #           Prediction(s) - {'virginica': '100 %'}
    |    #   #-> NO
    |    #       #-> Is petal_length >= 5.0?
    |    #       #-> YES
    |    #       #   #-> Is petal_width >= 1.6?
    |    #       #   #-> YES
    |    #       #   #   #-> Is petal_length >= 5.8?
    |    #       #   #   #-> YES
    |    #       #   #       Prediction(s) - {'virginica': '100 %'}
    |    #       #   #   #-> NO
    |    #       #   #       Prediction(s) - {'versicolor': '100 %'}
    |    #       #   #-> NO
    |    #       #       Prediction(s) - {'virginica': '100 %'}
    |    #       #-> NO
    |    #           #-> Is petal_width >= 1.7?
    |    #           #-> YES
    |    #               Prediction(s) - {'virginica': '100 %'}
    |    #           #-> NO
    |    #               Prediction(s) - {'versicolor': '100 %'}
    |    #-> NO
    |        Prediction(s) - {'setosa': '100 %'}

### Enhancements
Test decision tree after creation with testing data set ('classify' function already created).   
Calculate tree model efficiency with testing data.   
Optimize 'findBestQuestion' for large data sets and/or continuous values.   
Flake8 this script.

******

### Credits
##### https://www.youtube.com/watch?v=LDRbO9a6XPU&t=43s   
##### https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb   
