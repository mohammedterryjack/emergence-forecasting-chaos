from skmultilearn.adapt import MLkNN
from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#envs\myenv\Lib\site-packages\skmultilearn\adapt\mlknn.py
#self.knn_ = NearestNeighbors(n_neighbors=self.k).fit(X)

x = [
    [0,1,0,1],
    [1,1,0,1],
    [0,1,1,0]
]
y = [
    [1,0,0,1],
    [0,1,0,1],
    [0,0,1,1],        
]

x_train = x_test = x 
y_train = y_test = y

#model = KNeighborsClassifier(n_neighbors=3)
#model.fit(x_train, list(map(str,y_train)))
#predictions_new = model.predict(x_test)
#print(predictions_new)

model = MLkNN(k=2)

x_train = lil_matrix(x_train).toarray()
y_train = lil_matrix(y_train).toarray()
model.fit(x_train, y_train)

x_test = lil_matrix(x_test).toarray()

predictions_new = model.predict(x_test)
print(y_test)
print(predictions_new)
print("Accuracy = ",accuracy_score(y_test,predictions_new))


y_predictions = list(map(lambda prediction:prediction.toarray(),predictions_new))
print(y_predictions)
