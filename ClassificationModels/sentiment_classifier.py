from sentence_transformers import SentenceTransformer
import csv
from sklearn.tree import DecisionTreeClassifier
import pickle
		
clf = DecisionTreeClassifier(random_state=1)
data = []
X = []
Y = []
dictionary = {}
emotions_positivity_mapping = {
	"tone": False,
	"hopeful": True,
	"terrified": False,
	"sentimental": True,
	"trusting": True,
	"anticipating": True,
	"embarrassed": False,
	"guilty": False,
	"afraid": False,
	"sad": False,
	"surprised": False,
	"anxious": False,
	"lonely": False,
	"nostalgic": True,
	"confident": True,
	"impressed": True,
	"apprehensive": False,
	"disgusted": False,
	"excited": True,
	"furious": False,
	"content": True,
	"joyful": True,
	"angry": False,
	"prepared": False,
	"proud": True,
	"jealous": False,
	"devastated": False,
	"caring": True,
	"disappointed": False,
	"annoyed": False,
	"ashamed": False,
	"grateful": True,
	"faithful": False
	
	}

emotions_aggressive_mapping = {
	"tone": False,
	"hopeful": False,
	"terrified": False,
	"sentimental": False,
	"trusting": False,
	"anticipating": False,
	"embarrassed": False,
	"guilty": False,
	"afraid": False,
	"sad": False,
	"surprised": False,
	"anxious": False,
	"lonely": False,
	"nostalgic": False,
	"confident": False,
	"impressed": False,
	"apprehensive": False,
	"disgusted": True,
	"excited": False,
	"furious": True,
	"content": False,
	"joyful": False,
	"angry": True,
	"prepared": False,
	"proud": False,
	"jealous": False,
	"devastated": False,
	"caring": False,
	"disappointed": False,
	"annoyed": True,
	"ashamed": False,
	"grateful": True,
	"faithful": False
	
	}

with open('../DataSource/embedded_data.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		row[1] = emotions_aggressive_mapping[row[1]]
		data.append(row)
		
data = data[1::]

for datum in data:
	X.append(datum[2::])
	Y.append(datum[1])
	
	
clf.fit(X,Y)

with open('../PickleFiles/aggressive_sentiment.pickle', 'wb') as handle:
	pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)