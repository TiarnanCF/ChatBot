from sentence_transformers import SentenceTransformer
import csv
from sklearn.tree import DecisionTreeClassifier

model = SentenceTransformer("BAAI/bge-m3")
processed_data = []

with open('text_messages_tone.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		processed_data.append({"Message": row[0], "Tone": row[1], "Embedding": model.encode(row[0])})
		
csv_data = []

for datum in processed_data:
	csv_data.append([datum["Message"], datum["Tone"]] + datum["Embedding"].tolist())
	
with open('embedded_data.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	for row in csv_data:
		writer.writerow(row)