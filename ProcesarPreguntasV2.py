import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\ccast\Desktop\ClasificadorIA\clasificaciones_preguntas.xlsx')

preguntas = df['Pregunta'].tolist()
etiquetas = df['Clasificación'].tolist()

preguntas_entrenamiento = preguntas[:20]
etiquetas_entrenamiento = etiquetas[:20]

preguntas_no_etiquetadas = preguntas[20:]

vectorizer = TfidfVectorizer()
X_entrenamiento = vectorizer.fit_transform(preguntas_entrenamiento)

modelo = MultinomialNB()
modelo.fit(X_entrenamiento, etiquetas_entrenamiento)

X_no_etiquetadas = vectorizer.transform(preguntas_no_etiquetadas)
predicciones_no_etiquetadas = modelo.predict(X_no_etiquetadas)

df.loc[20:, 'Clasificación'] = predicciones_no_etiquetadas  

df.to_excel(r'C:\Users\ccast\Desktop\ClasificadorIA\clasificaciones_preguntas_completas.xlsx', index=False)

X_todas = vectorizer.transform(preguntas)
pca = PCA(n_components=2)
X_reducido = pca.fit_transform(X_todas.toarray())

colores = {'Positiva': 'g', 'Negativa': 'r'}
colores_preguntas = [colores[clasificacion] for clasificacion in df['Clasificación']]

plt.figure(figsize=(10, 8))
plt.scatter(X_reducido[:, 0], X_reducido[:, 1], c=colores_preguntas, marker='o')

for i, (x, y) in enumerate(X_reducido):
    plt.text(x, y, str(i + 1), fontsize=10)

plt.legend(['Positiva', 'Negativa'])
plt.title('Clasificación de preguntas sobre delincuencia')
plt.show()
