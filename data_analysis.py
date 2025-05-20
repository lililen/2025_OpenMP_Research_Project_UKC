import pandas as pd
import numpy as np

print("Environment working correctly!")
data = pd.DataFrame({"A": np.random.randint(1, 100, 5)})
print(data.head())