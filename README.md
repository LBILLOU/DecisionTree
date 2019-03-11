# Handmade Decision Tree :evergreen_tree: :rocket:

## Project Details
This short python script enables to quickly generate and visualize a decision tree generated from a training data set. Using dataframes from pandas, the scripts creates new branches by maximizing information gain from gini impurity calculations.

## Execution
    Git clone
    Python3 DecisionTree.py

## Example
    | ####   #####  #####  #  #####  #  #####  #    #     #####  ####   #####  #####
    | #   #  #      #      #  #      #  #   #  ##   #       #    #   #  #      #
    | #   #  ###    #      #  #####  #  #   #  # #  #       #    ###    ###    ###
    | #   #  #      #      #      #  #  #   #  #  # #       #    #  #   #      #
    | ####   #####  #####  #  #####  #  #####  #   ##       #    #   #  #####  #####
    |
    | >>> Import your csv file (filename.csv) :
    | sample.csv
    |
    | Imported Data Head
    |     color  size  label
    | 0   Green     3  Apple
    | 1  Yellow     3  Apple
    | 2     Red     1  Grape
    | 3     Red     1  Grape
    | 4  Yellow     3  Lemon
    | Imported Data Shape    (5, 3)
    | Imported Data Columns  ['color', 'size', 'label']
    |
    | >>> Which column from the above would you like to predict?
    | label
    |
    | #   #  #####  #    #  ####    #####  ####   #####  #####
    |  # #   #   #  #    #  #   #     #    #   #  #      #
    |   #    #   #  #    #  ###       #    ###    ###    ###
    |   #    #   #  #    #  #  #      #    #  #   #      #
    |   #    #####  ######  #   #     #    #   #  #####  #####
    |
    | Is size >= 3?
    | > YES
    |   Is color == Yellow?
    |   > YES
    |     Prediction(s) - {'Apple': '50 %', 'Lemon': '50 %'}
    |   > NO
    |     Prediction(s) - {'Apple': '100 %'}
    | > NO
    |   Prediction(s) - {'Grape': '100 %'}

### Enhancements
Test decision tree after creation with testing data set   
Calculate tree model efficiency   

******

#### Credits
##### https://www.youtube.com/watch?v=LDRbO9a6XPU&t=43s   
##### https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb   
