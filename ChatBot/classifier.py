from sentence_transformers import SentenceTransformer
import csv
from sklearn.tree import DecisionTreeClassifier
import pickle
		
clf = DecisionTreeClassifier(random_state=1)
data = []
X = []
Y = []


with open('embedded_data.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		data.append(row)
		
data = data[1::]

for datum in data:
	X.append(datum[2::])
	Y.append(datum[1])
	
	
clf.fit(X,Y)

with open('filename.pickle', 'wb') as handle:
	pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
