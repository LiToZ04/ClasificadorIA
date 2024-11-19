import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\ccast\Desktop\Codigos\Practicas\ClasificadorIA\Matriz.xlsx', header=None)
preguntas = df.iloc[0, 1:].dropna().tolist()

ejemplos_positivos = ["¿Qué acciones están tomando para reducir la delincuencia?", "¿Cómo ha mejorado la seguridad últimamente?"]
ejemplos_negativos = ["¿Por qué la delincuencia sigue aumentando?", "¿Por qué no hay seguridad en mi área?"]
ejemplos_neutrales = ["¿Cuáles son las políticas actuales de seguridad?", "¿Cómo se distribuye el presupuesto para seguridad?"]

corpus = ejemplos_positivos + ejemplos_negativos + ejemplos_neutrales + preguntas

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

num_etiquetadas = 20
clasificaciones = (
    ['Positiva'] * len(ejemplos_positivos) +
    ['Negativa'] * len(ejemplos_negativos) +
    ['Neutral'] * len(ejemplos_neutrales)
)

for i in range(num_etiquetadas):
    if i % 3 == 0:
        clasificaciones.append('Positiva')
    elif i % 3 == 1:
        clasificaciones.append('Negativa')
    else:
        clasificaciones.append('Neutral')

clasificaciones += ['Sin etiqueta'] * (len(preguntas) - num_etiquetadas)

clasificaciones_df = pd.DataFrame({
    'Pregunta': preguntas,
    'Clasificación': clasificaciones[len(ejemplos_positivos) + len(ejemplos_negativos) + len(ejemplos_neutrales):]
})
clasificaciones_df.to_excel(r'C:\Users\ccast\Desktop\Codigos\Practicas\ClasificadorIA\clasificaciones_preguntas.xlsx', index=False)

plt.figure(figsize=(10, 8))
colores = {'Positiva': 'g', 'Negativa': 'r', 'Neutral': 'b', 'Sin etiqueta': 'gray'}
colores_preguntas = [colores[clasificacion] for clasificacion in clasificaciones]

plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colores_preguntas, marker='o')

for i, pregunta in enumerate(preguntas[:num_etiquetadas]):
    plt.text(pca_result[i + len(ejemplos_positivos) + len(ejemplos_negativos) + len(ejemplos_neutrales), 0], 
             pca_result[i + len(ejemplos_positivos) + len(ejemplos_negativos) + len(ejemplos_neutrales), 1], 
             str(i + 1), fontsize=10)

plt.legend(['Positiva', 'Negativa', 'Neutral', 'Sin etiqueta'])
plt.title('Clasificación de preguntas sobre delincuencia (solo primeras 20 etiquetadas)')
plt.show()
