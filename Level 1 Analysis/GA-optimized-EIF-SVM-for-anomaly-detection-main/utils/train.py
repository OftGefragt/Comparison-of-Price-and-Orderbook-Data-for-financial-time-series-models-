from sklearn.preprocessing import MinMaxScaler
from eif.eif_class import iForest
import pickle
import pandas as pd

df = pd.read_csv('../data/train/BTC_1min.csv')
df = df.iloc[:, 1:-2]
df.head(2)
print(len(df.columns.tolist()))

scaler = MinMaxScaler()
X = scaler.fit_transform(df.values)

forest = iForest(X=X,
                 ntrees=235,
                 sample_size=1691,
                 ExtensionLevel=26)


scores_train = forest.compute_paths(X)

with open("eif_model.pkl", "wb") as f:
    pickle.dump(forest, f)


with open("eif_model.pkl", "wb") as f:
    pickle.dump(scaler, f)