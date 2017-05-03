import pandas as pd
import numpy as np
from math import log
from PIL import Image, ImageDraw
from sklearn.tree import tree


class LabelEncoder:
    def encode(self, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        return y

    def decode(self, y):
        return self.classes_[y]


label_encoder = LabelEncoder()


def read_data(path):
    data = pd.read_csv(path)
    data = data.drop("id", 1)
    # enc = LabelEncoder()
    # data[['type']] = enc.encode(data[['type']])
    y = data[['type']]
    X = data.drop('type', 1)
    y = label_encoder.encode(y)
    return X.as_matrix(), y, X.columns.values


def uniquecounts(y):
    result = []
    result.append(np.count_nonzero(y == 0))
    result.append(np.count_nonzero(y == 1))
    result.append(np.count_nonzero(y == 2))
    return result


def gini(y):
    total = len(y)
    counts = uniquecounts(y)
    imp = 0
    for i, k1 in enumerate(counts):
        if k1 == 0:
            continue
        p1 = float(counts[i]) / total
        for j, k2 in enumerate(counts):
            if k1 == k2: continue
            p2 = float(counts[j]) / total
            imp += p1 * p2
    return imp


def entropy(y):
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(y)
    ent = 0.0
    for i, r in enumerate(results):
        if r == 0:
            continue
        p = float(results[i]) / len(y)
        ent -= p * log2(p)
    return ent


class Node:
    def __init__(self, column=-1,
                 value=None,
                 true_branch=None, false_branch=None):
        self.column = column
        self.value = value
        # self.result_class = result_class
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree:
    def build(self, X, y, score=entropy):
        self.root = self.build_subtree(X, y, score)
        return self

    def build_subtree(self, X, y, score=entropy):
        if len(X) == 0:
            return Node()
        current_score = score(y)

        # Set up some variables to track the best criteria
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        column_count = len(X[0])
        for col in range(0, column_count):
            column_values = np.unique(X[:, col])

            for value in column_values:
                X1, y1, X2, y2 = self.divideset(X, y, col, value)

                # Information gain
                p = float(len(X1)) / len(X)
                gain = current_score - p * score(y1) - (1 - p) * score(y2)
                if gain > best_gain and len(X1) > 0 and len(X2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (X1, y1, X2, y2)
        # Create the sub branches
        if best_gain > 0:
            true_branch = self.build_subtree(best_sets[0], best_sets[1])
            false_branch = self.build_subtree(best_sets[2], best_sets[3])
            return Node(column=best_criteria[0], value=best_criteria[1],
                        true_branch=true_branch, false_branch=false_branch)
        else:
            # return Node(result_class=self.label_encoder.decode(np.argmax(uniquecounts(y))))
            return label_encoder.decode(np.argmax(uniquecounts(y)))

    def divideset(self, X, y, column, value):
        if isinstance(value, int) or isinstance(value, float):
            mask = X[:, column] >= value
        else:
            mask = X[:, column] == value

        return X[mask], y[mask], X[~mask], y[~mask]

    def predict(self, x):
        return self.classify_subtree(x, self.root)

    def classify_subtree(self, x, sub_tree):
        if not isinstance(sub_tree, Node):
            return sub_tree
        else:
            v = x[sub_tree.column]
            if isinstance(v, int) or isinstance(v, float):
                if v >= sub_tree.value:
                    branch = sub_tree.true_branch
                else:
                    branch = sub_tree.false_branch
            else:
                if v == sub_tree.value:
                    branch = sub_tree.true_branch
                else:
                    branch = sub_tree.false_branch
            return self.classify_subtree(x, branch)


def getwidth(tree):
    if not isinstance(tree, Node):
        return 1
    return getwidth(tree.true_branch) + getwidth(tree.false_branch)


def getdepth(tree):
    if not isinstance(tree, Node):
        return 0
    return max(getdepth(tree.true_branch), getdepth(tree.false_branch)) + 1


def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def drawnode(draw, tree, x, y):
    if isinstance(tree, Node):
        # Get the width of each branch
        shift = 100
        w1 = getwidth(tree.false_branch) * shift
        w2 = getwidth(tree.true_branch) * shift

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), columns[tree.column] + ':' + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + shift), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + shift), fill=(255, 0, 0))

        # Draw the branch nodes
        drawnode(draw, tree.false_branch, left + w1 / 2, y + shift)
        drawnode(draw, tree.true_branch, right - w2 / 2, y + shift)
    else:
        txt = tree
        draw.text((x - 20, y), txt, (0, 0, 0))


path = "halloween.csv"
X, y, columns = read_data(path)

tree1 = DecisionTree()
tree1 = tree1.build(X, y)
print(tree1.predict(X[0]))


drawtree(tree1.root)

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)


# with open("halloween.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f,
#                              feature_names=X.columns,
#                              class_names=['Ghoul', 'Goblin', 'Ghost'],
#                              )
#
# graph = pydotplus.graph_from_dot_file("halloween.dot")
# graph.write_pdf("halloween.pdf")
#
# data = pd.read_csv(path)
# data = data.drop("id", 1)
# data.to_csv("halloween.csv", index=False)
