import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.externals.six import StringIO
from sklearn import tree
import pydot

data = pd.read_csv('psych-redacted.csv')
independent_vars = data[data.columns[5:].values]
dependent_vars = data[data.columns[0]]
dependent_vars = [1 if entry =='Y' else 0 for entry in dependent_vars]

X_train,X_test,y_train,y_test = train_test_split(independent_vars,dependent_vars)

pipeline = Pipeline( [ ('clf',DecisionTreeClassifier(criterion='entropy') ) ] )

parameters = {
	'clf__max_depth':(150,155,160),
	'clf__min_samples_split':(1,2,3),
	'clf__min_samples_leaf':(1,2,3)
	}

grid_search = GridSearchCV(pipeline,parameters,n_jobs=-1,verbose=1,scoring='f1')
grid_search.fit(X_train,y_train)
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
	print '\t%s: %r' % (param_name, best_parameters[param_name])

predictions = grid_search.predict(X_test)
print classification_report(y_test,predictions)

'''
Function taken from:
http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

This function first starts with the nodes (identified by -1 in the child arrays)
 and then recursively finds the parents. I call this a node's 'lineage'.
  Along the way, I grab the values I need to create if/then/else SAS logic:
  
  .........
  
  The sets of tuples below contain everything I need to create SAS 
  if/then/else statements. I do not like using do blocks in SAS 
  which is why I create logic describing a node's entire path. The 
  single integer after the tuples is the ID of the terminal node in a path.
   All of the preceding tuples combine to create that node.
'''
def get_lineage(tree, feature_names):
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]     

     def recurse(left, right, child, lineage=None):          
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     for child in idx:
          for node in recurse(left, right, child):
               print node

clf = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes = 50)
clf.fit(X_train,y_train)

dot_data = StringIO()
tree.export_graphviz(clf,out_file = dot_data,feature_names=data.columns[5:].values)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('psych-entropy-decision-tree.pdf')

clf = DecisionTreeClassifier(max_leaf_nodes = 50)
clf.fit(X_train,y_train)
get_lineage(clf,data.columns[5:].values)
dot_data = StringIO()
tree.export_graphviz(clf,out_file = dot_data,feature_names=data.columns[5:].values)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('psych-50-node-gini-decision-tree.pdf')

clf = DecisionTreeClassifier(max_leaf_nodes = 50,max_depth=5)
clf.fit(X_train,y_train)
get_lineage(clf,data.columns[5:].values)
dot_data = StringIO()
tree.export_graphviz(clf,out_file = dot_data,feature_names=data.columns[5:].values)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('psych-5-depth-gini-decision-tree.pdf')

clf = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes = 50)
clf.fit(X_train,y_train)
get_lineage(clf,data.columns[5:].values)
dot_data = StringIO()
tree.export_graphviz(clf,out_file = dot_data,feature_names=data.columns[5:].values)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('psych-50-node-entropy-decision-tree.pdf')

mathdata = pd.read_csv('math-redacted.csv')
independent_vars = mathdata[mathdata.columns[6:].values]
dependent_vars = mathdata[mathdata.columns[0]]
dependent_vars = [1 if entry =='Y' else 0 for entry in dependent_vars]

X_train,X_test,y_train,y_test = train_test_split(independent_vars,dependent_vars)

clf = DecisionTreeClassifier(max_leaf_nodes = 50)
clf.fit(X_train,y_train)
get_lineage(clf,data.columns[5:].values)
dot_data = StringIO()
tree.export_graphviz(clf,out_file = dot_data,feature_names=data.columns[5:].values)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('math-50-node-gini-decision-tree.pdf')

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)
get_lineage(clf,data.columns[5:].values)
dot_data = StringIO()
tree.export_graphviz(clf,out_file = dot_data,feature_names=data.columns[5:].values)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('math-entropy-decision-tree.pdf')

clf = DecisionTreeClassifier(criterion='entropy',max_depth=5)
clf.fit(X_train,y_train)
get_lineage(clf,data.columns[5:].values)
dot_data = StringIO()
tree.export_graphviz(clf,out_file = dot_data,feature_names=data.columns[5:].values)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('math-5-depth-entropy-decision-tree.pdf')