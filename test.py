import pandas as pd

print ('--------init--------')

sampledf = pd.read_csv("fruits.csv", header=0)
targetcol = 2
targetcolname = 'label'
#train = sampledf[sampledf.columns.drop(targetcolname)]

def uniqueValues(dataf, col):
    return dataf[dataf.columns[col]].unique().tolist()

print(uniqueValues(sampledf, targetcol))

def countValues(dataf):
    output = {}
    for value in dataf[dataf.columns[targetcol]]:
        if value not in output:
            output[value] = 0
        output[value] += 1
    return output

# print countValues(sampledf)


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
        value = dataf[self.column]
        if isinstance(self.value, int) or isinstance(self.value, float):
            return value >= self.value
        else: return value == self.value


def branchsplit(dataf, question):
    print("branchsplit" + str(question))
    trueRows = pd.DataFrame(columns = dataf.columns.values)
    falseRows = pd.DataFrame(columns = dataf.columns.values)
    for index, row in sampledf.iterrows():
        if question.match(row):
            trueRows = pd.concat([trueRows, row.to_frame().transpose()])
        else: falseRows = pd.concat([falseRows, row.to_frame().transpose()])
    return trueRows, falseRows

def giniImp(dataf):
    count = countValues(dataf)
    impurity = 1
    for key in count:
        keyProb = float(count[key]) / dataf.shape[0]
        impurity -= keyProb**2
    return impurity

def infoGain(trueBrImp, falseBrImp, currentImp):
    p = float(len(trueBrImp)) / (len(trueBrImp) + len(falseBrImp))
    return currentImp - p * giniImp(trueBrImp) - (1 - p) * giniImp(falseBrImp)

def findBestQuestion(dataf):
    bestGain = 0
    bestQuestion = None
    currentImp = giniImp(dataf)
    colNumber = dataf.shape[1]

    for col in range(colNumber-1):
        if col != targetcol:
            values = uniqueValues(dataf, col)
            for val in values:
                question = Question(col, val)
                trueRows, falseRows = branchsplit(dataf, question)
                if trueRows.shape[0] == 0 or falseRows.shape[0] == 0:
                    # print 'aaa' ??????
                    continue
                gain = infoGain(trueRows, falseRows, currentImp)
                print(gain)
                if gain >= bestGain:
                    bestGain, bestQuestion = gain, question

    return bestGain, bestQuestion

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

print("aaaa")
my_tree = buildTree(sampledf)
print("bbb")
print(my_tree.question)
print(my_tree.trueBranch.predictions)
print(my_tree.falseBranch.predictions)

def printTree(node, spacing = ""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print (spacing + str(node.question))

    print (spacing + '--> True:')
    printTree(node.trueBranch, spacing + "  ")

    print (spacing + '--> False:')
    printTree(node.falseBranch, spacing + "  ")

print("***********************")
printTree(my_tree)
