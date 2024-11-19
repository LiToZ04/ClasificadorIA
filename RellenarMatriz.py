import pandas as pd

file_path = r'C:\Users\ccast\Desktop\Codigos\Practicas\ClasificadorIA\Matriz.xlsx'
df = pd.read_excel(file_path, sheet_name='Cuestionario unificado')

df_filled = df.apply(lambda col: col.fillna(col.mode()[0] if not col.mode().empty else 'No'))

df_filled = df_filled.replace({'Si': 1, 'No': 0})

df_filled.to_excel(r'C:\Users\ccast\Desktop\Codigos\Practicas\ClasificadorIA\MatrizLlena.xlsx', index=False)
print("Datos faltantes completados y archivo guardado como 'MatrizLlena.xlsx'")
