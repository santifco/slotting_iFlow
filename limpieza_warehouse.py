import pandas as pd
import numpy as np
import streamlit as st


st.set_page_config(page_title="Calculadora de Costo Total de Propiedad (TCO)", layout="wide")

@st.cache_data



def main_function():

    letras = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapeo_letras_a_numeros = {letra: i + 1 for i, letra in enumerate(letras)}
    print("Este script se ejecuta cuando se inicia el dashboard.")
    # Aquí puedes poner el código que necesites ejecutar
    datos_movimientos= pd.read_csv(r'E:\Mi unidad\iFlow\VKM - CDR\Ilolay\Movimientos\Septiembre\movimientos_ilolay_agosto_septiembre.csv',encoding='ISO-8859-1', sep=';')
    datos_articulos = pd.read_csv(r'E:\Mi unidad\iFlow\VKM - CDR\Ilolay\Maestro Articulos\Septiembre\articulos_sector_ilolay.csv',encoding='ISO-8859-1', sep=';')
    stock_articulos = pd.read_csv(r'E:\Mi unidad\iFlow\VKM - CDR\Ilolay\Posiciones\Septiembre\Posiciones_2024091195741.csv',encoding='ISO-8859-1', sep=';', skiprows=2,usecols=range(0, 14))
    datos = pd.read_csv(r'E:\Mi unidad\iFlow\VKM - CDR\Ilolay\Posiciones\Septiembre\Posiciones_2024091195741.csv',encoding='ISO-8859-1', sep=';', skiprows=2,usecols=range(0, 14))

    stock_articulos = stock_articulos[~stock_articulos['POSICION'].str.startswith('01MG')]
    stock_articulos = stock_articulos.dropna(subset=['ARTICULO'])
    stock_articulos['ARTICULO'] = stock_articulos['ARTICULO'].astype(str).str.strip()
    stock_articulos = stock_articulos[stock_articulos['ARTICULO'] != '']
    stock_articulos['ARTICULO'] = stock_articulos['ARTICULO'].astype(str).str.lstrip('0')
    stock_articulos = stock_articulos.rename(columns={'ARTICULO': 'Artículo'})
    stock_articulos['Artículo'] = stock_articulos['Artículo'].astype('int64')
    stock_articulos = stock_articulos.groupby(["Artículo"])["STOCK"].sum().reset_index()
    stock_articulos = stock_articulos.sort_values(by=['STOCK'], ascending=[False])

    datos_movimientos = datos_movimientos[~datos_movimientos['Artículo'].str.startswith('EMB')]
    datos_movimientos['Artículo'] = datos_movimientos['Artículo'].astype('int64')
    datos_movimientos['Gramos'] = datos_movimientos['Gramos'].str.lstrip('=')
    datos_movimientos['Gramos'] = datos_movimientos['Gramos'].apply(lambda x: float(eval(x)))
    datos_movimientos['Cantidad Blt'] = datos_movimientos['Cantidad Blt'].str.lstrip('=')
    datos_movimientos['Cantidad Blt'] = datos_movimientos['Cantidad Blt'].apply(lambda x: float(eval(x)))
    datos_movimientos["Kg"] = datos_movimientos['Gramos']/1000  
    datos_movimientos['Inicio'] = pd.to_datetime(datos_movimientos['Inicio'], format='%d/%m/%y %H:%M:%S')
    datos_movimientos['Fin'] = pd.to_datetime(datos_movimientos['Fin'], format='%d/%m/%y %H:%M:%S')
    datos_movimientos['Day'] = datos_movimientos['Fin'].dt.strftime('%d/%m/%y')
    datos_movimientos["Duracion"] = datos_movimientos["Fin"] - datos_movimientos["Inicio"]
    datos_movimientos['Duracion_horas'] = datos_movimientos["Duracion"].dt.total_seconds()/3600


    datos_movimientos = datos_movimientos.dropna(subset=['Posición O'])
    datos_movimientos = datos_movimientos.dropna(subset=['Posición D'])
    datos_movimientos['Sector_Pasillo'] = datos_movimientos['Posición O'].apply(lambda x: ''.join(c for c in x.split('-')[0].strip() if c.isalpha()))
    datos_movimientos['Columna'] = datos_movimientos['Posición O'].apply(lambda x: x.split('-')[1].strip())
    datos_movimientos['Sector_Pasillo_D'] = datos_movimientos['Posición D'].apply(lambda x: ''.join(c for c in x.split('-')[0].strip() if c.isalpha()))
    datos_movimientos['Columna_D'] = datos_movimientos['Posición D'].apply(lambda x: x.split('-')[1].strip())
    datos_movimientos = datos_movimientos.sort_values(by=['Sector_Pasillo', 'Columna'], ascending=[True, False])
    # Convertir las letras de cada valor en "Sector_Pasillo" a números y combinarlos
    datos_movimientos['Sector_Pasillo_Num'] = datos_movimientos['Sector_Pasillo'].apply(lambda x: sum(mapeo_letras_a_numeros[letra] * (26 ** i) for i, letra in enumerate(reversed(x))))
    datos_movimientos['Sector_Pasillo_Num'] = datos_movimientos['Sector_Pasillo_Num'].astype(int)
    datos_movimientos['Columna'] = datos_movimientos['Columna'].astype(int)
    datos_movimientos['Sector_Pasillo_Num_D'] = datos_movimientos['Sector_Pasillo_D'].apply(lambda x: sum(mapeo_letras_a_numeros[letra] * (26 ** i) for i, letra in enumerate(reversed(x))))
    datos_movimientos['Sector_Pasillo_Num_D'] = datos_movimientos['Sector_Pasillo_Num_D'].astype(int)
    datos_movimientos['Columna_D'] = datos_movimientos['Columna_D'].astype(int)
    

    datos = datos.dropna(subset=['POSICION'])
    datos = datos[~datos['POSICION'].str.contains('FIC|FF')]
    datos = datos[~datos['POSICION'].str.startswith('01MG')]
    datos = datos.rename(columns={'ARTICULO': 'Artículo'})
    datos['Artículo'] = datos['Artículo'].where(datos['FP']!= 0, 0)
    datos['FRAGIL'] = datos['Artículo'].where(datos['FP']!= 0, 1)
    datos['Artículo'] = datos['Artículo'].astype('int64')

    datos_movimientos_rp = datos_movimientos[datos_movimientos['Tipo'] == "RP"]
    cantidad_reposiciones = datos_movimientos_rp['Artículo'].value_counts().reset_index()
    cantidad_reposiciones.columns = ['Artículo', 'cantidad_repo']
    datos = pd.merge(datos, cantidad_reposiciones, on='Artículo', how='left')
    datos['cantidad_repo'] = datos['cantidad_repo'].fillna(0).astype(int)
    datos['cantidad_repo'] = datos['cantidad_repo']/20

    datos_movimientos_pi = datos_movimientos[datos_movimientos['Tipo'] == "PI"]
    cantidad_articulos = datos_movimientos_pi['Artículo'].value_counts().reset_index()
    cantidad_articulos.columns = ['Artículo', 'cantidad_movimientos']
    datos = pd.merge(datos, cantidad_articulos, on='Artículo', how='left')
    datos['cantidad_movimientos'] = datos['cantidad_movimientos'].fillna(0).astype(int)

    bultos_articulos = datos_movimientos.groupby('Artículo')['Cantidad'].sum().reset_index()
    bultos_articulos.columns = ['Artículo', 'cantidad_bultos']
    datos = pd.merge(datos, bultos_articulos, on='Artículo', how='left')
    datos['cantidad_bultos'] = datos['cantidad_bultos'].fillna(0).astype(int)
    datos["bultos/movimientos"] = datos['cantidad_bultos']/datos['cantidad_movimientos']
    datos['bultos/movimientos'] = datos['bultos/movimientos'].round(0)

    total_bultos_por_articulo = datos_movimientos.groupby('Artículo')['Cantidad'].sum().reset_index()
    total_bultos_por_articulo.columns = ['Artículo', 'Cantidad']
    datos = pd.merge(datos, total_bultos_por_articulo, on='Artículo', how='left')
    datos['Cantidad'] = datos['Cantidad'].fillna(0).astype(int)

    # Combinar tabla con pesos de los articulos
    datos_articulos = datos_articulos[~datos_articulos['Codigo Largo'].str.startswith('EMB')]
    datos_articulos = datos_articulos[datos_articulos['Codigo Largo'] != "VARIOS"]
    datos_articulos = datos_articulos[~datos_articulos['Codigo Largo'].isin(["Parlog","Pchep","Pplastico","Pavon2m","Pdescartable","testsistemas"])]
    datos_articulos = datos_articulos.dropna(subset=['Codigo Largo'])
    datos_articulos = datos_articulos.rename(columns={'Codigo Largo': 'Artículo'})
    datos_articulos['Artículo'] = datos_articulos['Artículo'].astype('int64')
    datos_articulos = datos_articulos[datos_articulos["FP"]==1]
    datos = pd.merge(datos, datos_articulos, on='Artículo', how='left')
    datos['Peso'] = datos['Peso'].round(1)
    # datos = pd.merge(datos, datos_articulos[['Artículo', 'Volumen']], on='Artículo', how='left')
    datos['Volumen'] = (datos['Volumen'].round(1))/1000000
    vol_promedio = datos['Volumen'].mean()
    datos['Volumen'] = datos['Volumen'].mask(datos['Volumen'] == 0, vol_promedio)
    datos["Densidad"] = datos['Peso']/datos['Volumen']

    # Extraer las partes relevantes de la posición
    datos['Sector_Pasillo'] = datos['POSICION'].apply(lambda x: ''.join(c for c in x.split('-')[0].strip() if c.isalpha()))
    datos['Columna'] = datos['POSICION'].apply(lambda x: x.split('-')[1].strip())
    datos = datos.sort_values(by=['Sector_Pasillo', 'Columna'], ascending=[True, False])
    # Convertir las letras de cada valor en "Sector_Pasillo" a números y combinarlos
    datos['Sector_Pasillo_Num'] = datos['Sector_Pasillo'].apply(lambda x: sum(mapeo_letras_a_numeros[letra] * (26 ** i) for i, letra in enumerate(reversed(x))))
    datos['Sector_Pasillo_Num'] = datos['Sector_Pasillo_Num'].astype(int)
    datos['Columna'] = datos['Columna'].astype(int)

    datos_movimientos = pd.merge(datos_movimientos, datos_articulos[['Artículo', 'Volumen']], on='Artículo', how='left')
    datos_movimientos["Volumen"] = (datos_movimientos["Volumen"]*datos_movimientos["Cantidad"])/1000000

    return datos_movimientos,datos_articulos,stock_articulos,datos

if __name__ == "__main__":
    main_function()
