import numpy as np
import pandas as pd

df = pd.read_csv("imports-85.csv").replace('?', np.nan)
