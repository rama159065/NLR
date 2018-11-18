print("=======================================> SAMIR")
print("Data=> ")
import pandas as pd
from core.algo.AlphaClassifierAlgorithm import AlphaClassifierAlgorithm
data = ["Adobe Inc."]
df = pd.DataFrame(data)
algorithm = AlphaClassifierAlgorithm()
print("Starting prediction of alphabetical attributes")
predictions = algorithm.predict(df)
print(predictions.head())
#label_df = get_labels(predictions)
#print(label_df)
