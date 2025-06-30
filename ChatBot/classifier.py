from sentence_transformers import SentenceTransformer
import csv
from sklearn.tree import DecisionTreeClassifier

model = SentenceTransformer("BAAI/bge-m3")

with open('text_messages_with_tone.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		synthetic_data.append({"Message": row[0], "Tone": row[1], "Embedding": model.encode(row[0])})
		
csv_data = []

for datum in synthetic_data:
	csv_data.append([datum["Message"], datum["Tone"]] + datum["Embedding"].tolist())
	
with open('embedded_data.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in csv_data:
		writer.writerow(row)
		
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