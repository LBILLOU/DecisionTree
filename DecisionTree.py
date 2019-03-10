import os
import pandas as pd

def getUserInput(question):
    # Funtion to retrieve input from user by asking an appropriate question
    print('')
    print(str(question))
    userInput = input()
    print('')
    return userInput

def uniqueValues(dataf, col):
    # Function to return unique values of a given column from a dataframe
    # using column index
    return dataf[dataf.columns[col]].unique().tolist()

def countValues(dataf, targetId):
    # Function to return the occurence of a value within a dataframe column
    output = {}
    for value in dataf[dataf.columns[targetId]]:
        if value not in output:
            output[value] = 0
        output[value] += 1
    return output

class Question():
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def __repr__(self):
        operator = ''
        if isinstance(self.value, int) or isinstance(self.value, float):
            operator = '>='
        else: operator = '=='
        return "Is column %s %s %s ?" % (self.column, operator, str(self.value))

#
#
#

    def match(self, dataf):
        value = dataf[list(dataf)[self.column]].values
        if isinstance(self.value, int) or isinstance(self.value, float):
            return value >= self.value
        else: return value == self.value # return type ndarray...
#
#
#

def branchsplit(dataf, question):
    trueRows = pd.DataFrame(columns = dataf.columns.values)
    falseRows = pd.DataFrame(columns = dataf.columns.values)
    #for rowNum in range(dataf.shape[0]):
    for rowNum in dataf.index.values:
        datafRow = dataf.loc[rowNum].to_frame().transpose()
        if question.match(datafRow):
            trueRows = pd.concat([trueRows, datafRow])
        else: falseRows = pd.concat([falseRows, datafRow])
    return trueRows, falseRows

def giniImp(dataf, targetId):
    dfValues = countValues(dataf, targetId)
    impurity = 1
    for key in dfValues:
        keyProb = float(dfValues[key]) / dataf.shape[0]
        impurity -= keyProb**2
    if impurity < 0: # to check if happens
        return print("Error, negative impurity")
    return impurity

def infoGain(left, right, currentImp, targetId):
    p = float(len(left)) / (len(left) + len(right))
    return currentImp - p * giniImp(left, targetId) - (1 - p) * giniImp(right, targetId)

def findBestQuestion(dataf, targetId):
    # Seeking best question through dataFrame
    bestGain = 0
    bestQuestion = None
    currentImp = giniImp(dataf, targetId)
    colNumber = dataf.shape[1]

    for col in range(colNumber):
        if col != targetId:
            values = uniqueValues(dataf, col)
            for val in values:
                question = Question(col, val)
                trueRows, falseRows = branchsplit(dataf, question)

                # Skipping branchsplit if it has not divided the dataset
                if trueRows.shape[0] == 0 and falseRows.shape[0] == 0:
                    continue

                # Calculating question gain
                gain = infoGain(trueRows, falseRows, currentImp, targetId)

                # Choosing question with the best gain
                if gain >= bestGain:
                    bestGain, bestQuestion = gain, question

    return bestGain, bestQuestion


class Leaf:
    def __init__(self, dataf, targetId):
        self.predictions = countValues(dataf, targetId)

class Node:
    def __init__(self, question, trueBranch, falseBranch):
        self.question = question
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch

def buildTree(dataf, targetId):
    gain, question = findBestQuestion(dataf, targetId)
    if gain == 0:
        return Leaf(dataf, targetId)
    trueRows, falseRows = branchsplit(dataf, question)
    trueBranch = buildTree(trueRows, targetId)
    falseBranch = buildTree(falseRows, targetId)
    return Node(question, trueBranch, falseBranch)

def printTree(node, spacing = ""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print (spacing + str(node.question))
    print (spacing + '--> True:')
    printTree(node.trueBranch, spacing + "  ")
    print (spacing + '--> False:')
    printTree(node.falseBranch, spacing + "  ")

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.trueBranch)
    else:
        return classify(row, node.falseBranch)

def printLeaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

# MAIN

def main():
    print('--------------------------------')
    print('--- DECISION TREE VISUALIZER ---')
    print('--------------------------------')

    # csv file selection by user
    filename = getUserInput('>>> Import your csv file (filename.csv) :')
    while os.path.isfile(filename) is False:
        filename = getUserInput('>>> Error, file not found. Please try again :')

    # filename = "fruits.csv" # for testing
    impCSV = pd.read_csv(filename, header=0)

    print('Imported Data Shape : ', end = '')
    print(impCSV.shape)

    print('Imported Data columns : ', end = '')
    impColNames = list(impCSV.columns)
    print(impColNames)

    userTarget = getUserInput('>>> Which column from the above would you like to predict?')
    while userTarget not in impColNames:
        userTarget = getUserInput('>>> Error, Please choose a column you like to predict from the following list: ' + str(print(impColNames)))

    # target = impCSV[userTarget]

    target = 'label'
    targetId = impCSV.columns.get_loc(target)

    print("> Function *** uniqueValues")
    print("out --> " + str(uniqueValues(impCSV, targetId)))
    print(type(uniqueValues(impCSV, targetId)))

    print("> Function *** countValues")
    print("out --> " + str(countValues(impCSV, targetId)))
    print(type(countValues(impCSV, targetId)))

    print("> Class *** Question")
    q = Question(0, 'Green')
    print(q)
    row = impCSV.loc[2, :].to_frame().transpose()

    print("> Function *** Question.match")
    print("out --> " + str(q.match(row)))
    print(type(q.match(row)))

    #for i in impCSV:
    print(impCSV.loc[1].to_frame().transpose())

    print("> Function *** branchsplit")
    true , false = branchsplit(impCSV, q)
    print("out --> " )
    print(true)
    print(false)
    print(type(true))

    print("> Function *** giniImp")
    print("out --> " + str(giniImp(false, targetId)))
    print(type(giniImp(true, targetId)))

    print("> Function *** infoGain")
    out = infoGain(true, false, giniImp(impCSV, targetId), targetId)
    print("out --> " + str(out))
    print(type(out))

    print("> Function *** findBestQuestion")
    out = findBestQuestion(impCSV, targetId)
    print("out --> " + str(out))
    print(type(out))

    print("> Function *** buildTree")
    my_tree = buildTree(impCSV, targetId)
    print("out --> " + str(out))
    print(type(out))

    print("> Function *** printTree")
    printTree(my_tree)

    print("> Function *** testTree")
    test = classify(impCSV.loc[2,:].to_frame().transpose(), my_tree)
    print("out --> " + str(test))
    print(type(test))

    filename2 = "testingFruits.csv" # for testing
    impCSV2 = pd.read_csv(filename, header=0)
    for index, row in impCSV2.iterrows():
        print ("Actual: %s. Predicted: %s " % (row[targetId], printLeaf(classify(row.to_frame().transpose(), my_tree))))


main()
