import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel(r'C:\Users\ccast\Desktop\Codigos\Practicas\ClasificadorIA\MatrizLlena.xlsx')

data['Suma Respuestas'] = data.iloc[:, 1:].apply(lambda row: row[(row == 1) | (row == 0)].sum(), axis=1)
data['Etiqueta'] = np.where(data['Suma Respuestas'] > 100, 1, 
                            np.where(data['Suma Respuestas'] < 90, -1, 0))

X = data[['Suma Respuestas']]
y = data['Etiqueta']

data['Edad'] = np.random.randint(18, 61, size=len(data))

plt.figure(figsize=(10, 6))
plt.scatter(data['Suma Respuestas'], data['Edad'], c=data['Etiqueta'], cmap='coolwarm', alpha=0.6, edgecolors='k')
plt.axvline(x=90, color='red', linestyle='--', label='Límite inseguro (Suma < 90)')
plt.axvline(x=100, color='green', linestyle='--', label='Límite seguro (Suma > 100)')
plt.title('Distribución de Puntajes de Seguridad por Edad')
plt.xlabel('Suma de Respuestas (0-150)')
plt.ylabel('Edad (18-60)')
plt.colorbar(label='Clasificación (-1: Inseguro, 0: Neutral, 1: Seguro)')
plt.legend()
plt.grid()
plt.show()

with pd.ExcelWriter(r'C:\Users\ccast\Desktop\Codigos\Practicas\ClasificadorIA\MatrizConEtiquetas.xlsx') as writer:
    data.to_excel(writer, sheet_name='Datos', index=False)
    conteo = data['Etiqueta'].value_counts().rename_axis('Etiqueta').reset_index(name='Cantidad')
    conteo['Etiqueta'] = conteo['Etiqueta'].map({1: 'Seguro', 0: 'Neutral', -1: 'Inseguro'})
    conteo.to_excel(writer, sheet_name='Conteo', index=False)

print(data)
