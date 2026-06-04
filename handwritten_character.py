from sklearn.datasets import load_digits
digits =load_digits()
print("Total Images:",len(digits.images))
print("First Label:",digits.target[0])
from sklearn.model_selection import train_test_split
x=digits.data
y=digits.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Training Data:",len(x_train))
print("Testing Data:",len(x_test))
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
print("Model Trained Successfully")
from sklearn.metrics import accuracy_score
y_pred =model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(cm)
sample=[x_test[0]]
prediction=model.predict(sample)
print("predicted digit:",prediction[0])
print("Actual Digit:",y_test.iloc[0] if hasattr(y_test,'iloc') else y_test[0])
print("handwritten character recognition project completed successfully")
print("algorithm used:Random Forest")
print("Accuracy:",accuracy)
