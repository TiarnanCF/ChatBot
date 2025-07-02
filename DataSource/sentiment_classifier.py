from sentence_transformers import SentenceTransformer
import csv
from sklearn.tree import DecisionTreeClassifier
import pickle
		
clf = DecisionTreeClassifier(random_state=1)
data = []
X = []
Y = []
dictionary = {}
emotions_mapping = {
	"tone": "N/A",
	"hopeful": "Positive",
	"terrified": "Negative",
	"sentimental": "Positive",
	"trusting": "Positive",
	"anticipating": "Positive",
	"embarrassed": "Negative",
	"guilty": "Negative",
	"afraid": "Negative",
	"sad": "Negative",
	"surprised": "Neutral",
	"anxious": "Negative",
	"lonely": "Negative",
	"nostalgic": "Positive",
	"confident": "Positive",
	"impressed": "Positive",
	"apprehensive": "Negative",
	"disgusted": "Negative",
	"excited": "Positive",
	"furious": "Negative",
	"content": "Positive",
	"joyful": "Positive",
	"angry": "Negative",
	"prepared": "Neutral",
	"proud": "Positive",
	"jealous": "Negative",
	"devastated": "Negative",
	"caring": "Positive",
	"disappointed": "Negative",
	"annoyed": "Negative",
	"ashamed": "Negative",
	"grateful": "Positive",
	"faithful": "Neutral"
	
	}

with open('embedded_data.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		row[1] = emotions_mapping[row[1]]
		data.append(row)
		
data = data[1::]

for datum in data:
	X.append(datum[2::])
	Y.append(datum[1])
	
	
clf.fit(X,Y)

with open('filename_2.pickle', 'wb') as handle:
	pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)