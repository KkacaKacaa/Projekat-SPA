# Klasifikacija teksta
Ovaj projekat pomaže u klasifikovanju teksta koji može da razlikuje različite kategorije tekstova.
Glavni paketi koji se koriste u ovom projektu su: sklearn, pandas, dataset i spacy.

## Instalacija
Potrebno je importovati pomenute datoteke pomoću package-manager-a [pip].

```bash
!pip install pandas
!pip install scikit-learn
```

## Importovanje potrebnih biblioteka

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import spacy
```
## Čitanje datoteke 'dataset.csv'

```python
fajl = pd.read_csv('dataset.csv')
X = fajl['Message']
y = fajl['Category']
```
## Treniranje i testiranje podataka

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Pravljenje klasifikatora

```python
pipe_MNB = Pipeline([
('tfidf', TfidfVectorizer()),('clf', MultinomialNB())
])
pipe_CNB = Pipeline([
('tfidf', TfidfVectorizer()),('clf', ComplementNB())
])
pipe_SVC = Pipeline([
('tfidf', TfidfVectorizer()),('clf', LinearSVC())
])
```
## Građenje modela koristeći MultinomialNB, ComplementNB i LinearSVC, i obučavanje

```python
pipe_MNB.fit(X_train, y_train)
predict_MNB = pipe_MNB.predict(X_test)
print(f"MNB: {accuracy_score(y_test, predict_MNB):.2f}")

pipe_CNB.fit(X_train, y_train)
predict_CNB = pipe_CNB.predict(X_test)
print(f"CNB: {accuracy_score(y_test, predict_CNB):.2f}")

pipe_SVC.fit(X_train, y_train)
predict_SVC = pipe_SVC.predict(X_test)
print(f"SVC: {accuracy_score(y_test, predict_SVC):.2f}")
```
## Testiranje

```python
message = "you have won a $10000 prize! contact us fot eh reward!"
result = pipe_SVC.predict([message])
print("Result: ", result[0])
```
