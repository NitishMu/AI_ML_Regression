import pandas as pd 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
#
iris=datasets.load_iris()
x=pd.DataFrame(iris.data,columns=iris.feature_names)
y=iris.target 
#
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#
clf=DecisionTreeClassifier(random_state=42)
#
clf.fit(x_train,y_train)
#
y_pred=clf.predict(x_test)
#
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy}")
#
new_sample=[[5.1,3.5,1.4,0.2]]
#
predicted_species=iris.target_names[clf.predict(new_sample)[0]]
print(f"Predicted species:{predicted_species}") 
