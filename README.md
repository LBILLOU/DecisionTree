# Homemade Decision Tree

## Execution
    Git clone
    Run DecisionTree.py with python 3.6.5

## Project Details

This short python script enables to quickly generate and visualize a decision tree generated from a training data set. Using dataframes from pandas, the scripts creates new branches by maximizing information gain from gini impurity calculations.

## Example

Example using sample.csv ::

  & ####   #####  #####  #  #####  #  #####  #    #     #####  ####   #####  #####
  & #   #  #      #      #  #      #  #   #  ##   #       #    #   #  #      #
  & #   #  ###    #      #  #####  #  #   #  # #  #       #    ###    ###    ###
  & #   #  #      #      #      #  #  #   #  #  # #       #    #  #   #      #
  & ####   #####  #####  #  #####  #  #####  #   ##       #    #   #  #####  #####
  &
  & >>> Import your csv file (filename.csv) :
  & sample.csv
  &
  & Imported Data Head
  &     color  size  label
  & 0   Green     3  Apple
  & 1  Yellow     3  Apple
  & 2     Red     1  Grape
  & 3     Red     1  Grape
  & 4  Yellow     3  Lemon
  & Imported Data Shape    (5, 3)
  & Imported Data Columns  ['color', 'size', 'label']
  &
  & >>> Which column from the above would you like to predict?
  & label
  &
  & #   #  #####  #    #  ####    #####  ####   #####  #####
  &  # #   #   #  #    #  #   #     #    #   #  #      #
  &   #    #   #  #    #  ###       #    ###    ###    ###
  &   #    #   #  #    #  #  #      #    #  #   #      #
  &   #    #####  ######  #   #     #    #   #  #####  #####
  &
  & Is size >= 3?
  & > YES
  &   Is color == Yellow?
  &   > YES
  &     Prediction(s) - {'Apple': '50 %', 'Lemon': '50 %'}
  &   > NO
  &     Prediction(s) - {'Apple': '100 %'}
  & > NO
  &   Prediction(s) - {'Grape': '100 %'}

###### Credits
* https://www.youtube.com/watch?v=LDRbO9a6XPU&t=43s
* https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
