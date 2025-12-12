# %%
import pandas as pd

df = pd.read_csv('../data/abt_churn.csv')
df.head()

# %%
oot = df[df['dtRef']==df['dtRef'].max()].copy()

# %%
df_train = df[df['dtRef']<df['dtRef'].max()].copy()

# %%
features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]

# %%
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y,
)

# %%
print(f'Taxa da váriavel resposta GERAL: {y.mean()}')
print(f'Taxa da váriavel resposta TRAIN: {y_train.mean()}')
print(F'Taxa da váriavel resposta TEST: {y_test.mean()}')

# %%
X_train.isna().sum().sort_values(ascending=False)

# %%
