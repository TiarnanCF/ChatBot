import random
import csv
from datasets import load_dataset

# 1) Load datasets
#dd = load_dataset("daily_dialog", split="train")                   # ~87k utterances
ed = load_dataset("empathetic_dialogues", split="train")           # ~24k conversations

# 2) Helper: map DailyDialog emotions → tones
DD_MAP = {
    0: "Angry",  # anger
    1: "Angry",  # disgust
    2: "Neutral",# fear
    3: "Excited",# happiness
    4: "Apologetic",# sadness
    5: "Excited",# surprise
    6: "Neutral" # neutral
}

# 3) Helper: map EmpatheticDialogues emotions → tones
ED_MAP = {
    "proud":       "Excited",
    "joyful":      "Excited",
    "excited":     "Excited",
    "amused":      "Excited",
    "embarrassed": "Apologetic",
    "apologetic":  "Apologetic",
    "regretful":   "Apologetic",
    "annoyed":     "Angry",
    "angry":       "Angry",
    "frustrated":  "Angry",
    "hopeless":    "Apologetic",
    "sad":         "Apologetic",
    "disappointed":"Apologetic",
    "grateful":    "Friendly",
    "trustful":    "Friendly",
    "caring":      "Friendly",
    # any others default to "Neutral"
}

# 4) Pull utterances + map labels
records = []

# From DailyDialog
#for ex in dd:
#    for key in ex:
#        print(key)
#    #text = ex["dialog"][-1]  # last utterance in the turn
#    #tone = DD_MAP.get(ex["emotion"], "Neutral")
#    #records.append((text, tone))

# From EmpatheticDialogues
for conv in ed:
    emo = conv["context"]
    tone = emo#ED_MAP.get(emo, "Neutral")
    text = conv["utterance"]
    records.append((text, tone))

# 5) Filter unique & shuffle
unique = list({(txt.strip(), ton) for txt, ton in records if len(txt.strip())>5})
random.shuffle(unique)

# 7) Write CSV
with open("text_messages_tone.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["message", "tone"])
    writer.writerows(unique)

print(f"Saved {len(unique)} messages → text_messages_tone_10000.csv")