import numpy as np
import pandas as pd

column_names = [
    "class", "age", "sex", "steroid", "antivirals", "fatigue", "malaise"
    "anorexia", "liver_big", "liver_firm", "spleen_palpable", "spiders",
    "ascites", "varices", "bilirubin", "alk_phosphate", "sgot", "albumin",
    "protime", "histology"
]

df = pd.read_csv("hepatitis.csv", names=column_names)\
    .replace("?", np.nan)\
    .apply(pd.to_numeric)\
    .drop_duplicates()

df = df.fillna(df.mean().to_dict())

print(df)
