import pandas as pd

column_names = [
    "class", "largest_spot_size", "spot_distribution", "activity",
    "evolution", "past24hour_flare_activity", "historically_complex",
    "region_became_historically_complex_on_pass", "area", "largest_spot_area",
    "common_flare", "moderate_flare", "severe_flare"
]

flare_1 = pd.read_csv("flare1.csv", sep=" ", names=column_names)
flare_2 = pd.read_csv("flare2.csv", sep=" ", names=column_names)

df = pd.concat([flare_1, flare_2])
