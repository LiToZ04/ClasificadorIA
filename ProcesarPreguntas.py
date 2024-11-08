import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\ccast\Desktop\IA_V4S\Matriz.xlsx', header=None)
preguntas = df.iloc[0, 1:].dropna().tolist()


ejemplos_positivos = ["¿Qué acciones están tomando para reducir la delincuencia?", "¿Cómo ha mejorado la seguridad últimamente?"]
ejemplos_negativos = ["¿Por qué la delincuencia sigue aumentando?", "¿Por qué no hay seguridad en mi área?"]

corpus = ejemplos_positivos + ejemplos_negativos + preguntas

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

num_etiquetadas = 20
clasificaciones = ['Positiva'] * len(ejemplos_positivos) + ['Negativa'] * len(ejemplos_negativos)
clasificaciones += ['Positiva' if i % 2 == 0 else 'Negativa' for i in range(num_etiquetadas)]
clasificaciones += ['Sin etiqueta'] * (len(preguntas) - num_etiquetadas)

clasificaciones_df = pd.DataFrame({
    'Pregunta': preguntas,
    'Clasificación': clasificaciones[len(ejemplos_positivos) + len(ejemplos_negativos):]
})
clasificaciones_df.to_excel(r'C:\Users\ccast\Desktop\IA_V4S\clasificaciones_preguntas.xlsx', index=False)

plt.figure(figsize=(10, 8))
colores = {'Positiva': 'g', 'Negativa': 'r', 'Sin etiqueta': 'gray'}
colores_preguntas = [colores[clasificacion] for clasificacion in clasificaciones]

plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colores_preguntas, marker='o')

for i, pregunta in enumerate(preguntas[:num_etiquetadas]):
    plt.text(pca_result[i + len(ejemplos_positivos) + len(ejemplos_negativos), 0], 
            pca_result[i + len(ejemplos_positivos) + len(ejemplos_negativos), 1], 
            str(i + 1), fontsize=10)

plt.legend(['Positiva', 'Negativa', 'Sin etiqueta'])
plt.title('Clasificación de preguntas sobre delincuencia (solo primeras 20 etiquetadas)')
plt.show()


#clasificación basada en similitud de coseno es semi-supervisado pq necesita ejemplos.

# Los ejes X y Y en el gráfico no tienen un significado directo en términos de palabras o temas específicos.
# En cambio, lo que representan son **combinaciones lineales de las características** de las preguntas 
# que mejor capturan y representan la variabilidad de los datos.
#
# Estas combinaciones lineales son generadas por el algoritmo PCA, que busca reducir las dimensiones de los datos
# preservando la mayor parte de la información posible. En otras palabras, cada eje en el gráfico representa 
# una "combinación" de las características de las preguntas que permite separar o diferenciar las preguntas 
# en función de sus términos.
#
# Por ejemplo, si una pregunta tiene un alto valor para ciertas palabras clave y otra pregunta tiene un 
# alto valor para un conjunto diferente de palabras clave, PCA ajustará las coordenadas X e Y para maximizar 
# la distancia entre ellas en el nuevo espacio 2D, reflejando así sus diferencias en el contenido semántico.
#
# Esto significa que, aunque los ejes no tienen una interpretación directa de palabras, las distancias entre 
# los puntos en el gráfico reflejan la similitud o diferencia de los temas y términos entre las preguntas.
