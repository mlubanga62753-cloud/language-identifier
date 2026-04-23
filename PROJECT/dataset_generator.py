# dataset_generator.py
import pandas as pd
import random

swahili_base = [
    "Niko soko leo", "Ninakupenda sana", "Tutakutana kesho",
    "Habari yako rafiki", "Naenda nyumbani", "Leo ni siku nzuri",
    "Nahitaji msaada wako", "Nimechoka leo"
]

english_base = [
    "I am going home", "Let us meet tomorrow", "This is a good day",
    "She likes reading books", "We are learning NLP", "I love programming",
    "He is my friend", "The weather is nice today"
]

sheng_base = [
    "Wacha tuende base", "Niko rada leo", "Uko aje buda",
    "Hii manzi ni fiti", "Tuko street vibaya", "Basi manze relax",
    "Nimechill na boys", "Hii life ni tricky"
]

luhya_base = [
    "Ndi khwitsa eshikulu", "Olaba otya", "Khuli omundu mulayi",
    "Wabula shani", "Nakhubona khulayi", "Khukhala bulayi"
]

prefix = ["yo", "hey", "", "bro", "man"]
suffix = ["bro", "boss", "man", "", ""]

def augment(sentence):
    return f"{random.choice(prefix)} {sentence} {random.choice(suffix)}".strip()

def generate(n=300):
    data = []

    for _ in range(n):
        data.append((augment(random.choice(swahili_base)), "Swahili"))
        data.append((augment(random.choice(english_base)), "English"))
        data.append((augment(random.choice(sheng_base)), "Sheng"))
        data.append((augment(random.choice(luhya_base)), "Luhya"))

    df = pd.DataFrame(data, columns=["text", "language"])
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv("data/dataset.csv", index=False)
    print("Dataset created:", len(df), "samples")

generate(300)  # 1200 samples total