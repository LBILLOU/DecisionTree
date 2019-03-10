import os
import pandas as pd

print ('--------init--------')

# csv file selection by user
"""print ('>>> Import data from csv file, which file? (filename.csv)')
filename = input()
while os.path.isfile(filename) is False:
    print('ERROR -> Please choose a csv file in current directory.')
    filename = input()"""

filename = "fruits.csv" # for testing
impCSV = pd.read_csv(filename, header=0)

print('Imported Data Shape')
print(impCSV.shape)

print('Imported Data columns')
impColNames = list(impCSV.columns)
print(impColNames)

"""print('>>> Which column would you like to predict?')
userTarget = input()
while userTarget not in impColNames:
    print('ERROR -> Please choose a column you like to predict from the following list:')
    print(impColNames)
    userTarget = input()

target = impCSV[userTarget] """
target = 'label'
targetId = impCSV.columns.get_loc(target)

def uniqueValues(dataf, col):
    return dataf[dataf.columns[col]].unique().tolist()

print("> Function *** uniqueValues")
print("out --> " + str(uniqueValues(impCSV, targetId)))
print(type(uniqueValues(impCSV, targetId)))

def countValues(dataf):
    output = {}
    for value in dataf[dataf.columns[targetId]]:
        if value not in output:
            output[value] = 0
        output[value] += 1
    return output

print("> Function *** countValues")
print("out --> " + str(countValues(impCSV)))
print(type(countValues(impCSV)))

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

    def match(self, dataf):
        value = dataf[list(dataf)[self.column]].values
        if isinstance(self.value, int) or isinstance(self.value, float):
            return value >= self.value
        else: return value == self.value # return type ndarray...

print("> Class *** Question")
q = Question(0, 'Green')
print(q)
row = impCSV.loc[2, :].to_frame().transpose()
# print(row)
print("> Function *** Question.match")
print("out --> " + str(q.match(row)))
print(type(q.match(row)))

#for i in impCSV:
print(impCSV.loc[1].to_frame().transpose())

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

print("> Function *** branchsplit")
true , false = branchsplit(impCSV, q)
print("out --> " )
print(true)
print(false)
print(type(true))

def giniImp(dataf):
    dfValues = countValues(dataf)
    impurity = 1
    for key in dfValues:
        keyProb = float(dfValues[key]) / dataf.shape[0]
        impurity -= keyProb**2
    if impurity < 0: # to check if happens
        return print("Error, negative impurity")
    return impurity

print("> Function *** giniImp")
print("out --> " + str(giniImp(false)))
print(type(giniImp(true)))

def infoGain(left, right, currentImp):
    p = float(len(left)) / (len(left) + len(right))
    return currentImp - p * giniImp(left) - (1 - p) * giniImp(right)

print("> Function *** infoGain")
out = infoGain(true, false, giniImp(impCSV))
print("out --> " + str(out))
print(type(out))

def findBestQuestion(dataf):
    # Seeking best question through dataFrame
    bestGain = 0
    bestQuestion = None
    currentImp = giniImp(dataf)
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
                gain = infoGain(trueRows, falseRows, currentImp)

                # Choosing question with the best gain
                if gain >= bestGain:
                    bestGain, bestQuestion = gain, question

    return bestGain, bestQuestion

print("> Function *** findBestQuestion")
out = findBestQuestion(impCSV)
print("out --> " + str(out))
print(type(out))

class Leaf:
    def __init__(self, dataf):
        self.predictions = countValues(dataf)

class Node:
    def __init__(self, question, trueBranch, falseBranch):
        self.question = question
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch

def buildTree(dataf):
    gain, question = findBestQuestion(dataf)
    if gain == 0:
        return Leaf(dataf)
    trueRows, falseRows = branchsplit(dataf, question)
    trueBranch = buildTree(trueRows)
    falseBranch = buildTree(falseRows)
    return Node(question, trueBranch, falseBranch)

print("> Function *** buildTree")
my_tree = buildTree(impCSV)
print("out --> " + str(out))
print(type(out))

def printTree(node, spacing = ""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print (spacing + str(node.question))
    print (spacing + '--> True:')
    printTree(node.trueBranch, spacing + "  ")
    print (spacing + '--> False:')
    printTree(node.falseBranch, spacing + "  ")

print("> Function *** printTree")
printTree(my_tree)

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.trueBranch)
    else:
        return classify(row, node.falseBranch)

print("> Function *** testTree")
test = classify(impCSV.loc[2,:].to_frame().transpose(), my_tree)
print("out --> " + str(test))
print(type(test))

def printLeaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

filename2 = "testingFruits.csv" # for testing
impCSV2 = pd.read_csv(filename, header=0)
for index, row in impCSV2.iterrows():
    print ("Actual: %s. Predicted: %s " % (row[targetId], printLeaf(classify(row.to_frame().transpose(), my_tree))))
