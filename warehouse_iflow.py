import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import geopandas as gpd

st.set_page_config(page_title="Calculadora de Costo Total de Propiedad (TCO)", layout="wide")

with st.expander("Carga de archivos"):

    
    datos_movimientos = st.file_uploader("Movimientos", type="csv")
    datos_articulos = st.file_uploader("Artículos", type="csv")
    posiciones = st.file_uploader("Posiciones", type="csv")
    
    letras = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapeo_letras_a_numeros = {letra: i + 1 for i, letra in enumerate(letras)}
    print("Este script se ejecuta cuando se inicia el dashboard.")
    # Aquí puedes poner el código que necesites ejecutar
    datos = pd.read_csv(posiciones,encoding='ISO-8859-1', sep=';',skiprows=2)

    datos_movimientos= pd.read_csv(datos_movimientos,encoding='ISO-8859-1', sep=';')
    datos_articulos = pd.read_csv(datos_articulos,encoding='ISO-8859-1', sep=';')
    # stock_articulos = pd.read_csv(posiciones,encoding='ISO-8859-1', sep=';', skiprows=2,usecols=range(0, 14))

    # stock_articulos = stock_articulos[~stock_articulos['POSICION'].str.startswith('01MG')]
    # stock_articulos = stock_articulos.dropna(subset=['ARTICULO'])
    # stock_articulos['ARTICULO'] = stock_articulos['ARTICULO'].astype(str).str.strip()
    # stock_articulos = stock_articulos[stock_articulos['ARTICULO'] != '']
    # stock_articulos['ARTICULO'] = stock_articulos['ARTICULO'].astype(str).str.lstrip('0')
    # stock_articulos = stock_articulos.rename(columns={'ARTICULO': 'Artículo'})
    # stock_articulos['Artículo'] = stock_articulos['Artículo'].astype('int64')
    # stock_articulos = stock_articulos.groupby(["Artículo"])["STOCK"].sum().reset_index()
    # stock_articulos = stock_articulos.sort_values(by=['STOCK'], ascending=[False])

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




# datos_movimientos = main_function()[0]
# datos_articulos = main_function()[1]
# stock_articulos = main_function()[2]
# datos = main_function()[3]


# Ejecutar el script al iniciar el dashboard
st.write("Ejecutando script al iniciar el dashboard...")


# Título del dashboard
st.title('Mapa interactivo iFlow ')

# Descripción
st.write("""
Este es un ejemplo del Fleet paralelo interactivo creado con Streamlit.
""")

# Leer los datos desde el archivo Excel



with st.sidebar:

    # fecha_seleccionados = st.multiselect('Fecha:', fechas_unicas, default=["Todas las Fechas"])

    # Crear un widget de selección de fecha
    fecha_inicio = st.date_input('Fecha inicial:', value=datos_movimientos['Inicio'].min())
    fecha_fin = st.date_input('Fecha final:', value=datos_movimientos['Inicio'].max())



fragil = 1
lista_valores = [0,2032629,2032524,2033033]
limite_densidad = 20000000000
reemplazos = []
limites_origin = []
primeros = []
direccion_livianos = "horizontal"



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

fecha_inicio_dt = pd.to_datetime(fecha_inicio)
fecha_fin_dt = pd.to_datetime(fecha_fin)

datos_movimientos = datos_movimientos[(datos_movimientos['Inicio'] >= fecha_inicio_dt) & (datos_movimientos['Inicio'] <= fecha_fin_dt)]

#Limpiar los datos

tipo_movimiento = "PI"
datos_movimientos = datos_movimientos[datos_movimientos['Tipo'] == tipo_movimiento]




# Crear un diccionario para mapear letras a números basados en su orden alfabético
letras = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'   
mapeo_letras_a_numeros = {letra: i + 1 for i, letra in enumerate(letras)}


almacen = datos[['Sector_Pasillo_Num',"Columna"]]

cantidad_pasillos = datos['Sector_Pasillo'].nunique()
# print(f"cantidad pasillos: {cantidad_pasillos}")

# datos_movimientos['Sector_Pasillo'] = datos_movimientos['Posición O'].apply(lambda x: ''.join(c for c in x.split('-')[0].strip() if c.isalpha()))
# datos_movimientos['Columna'] = datos_movimientos['Posición O'].apply(lambda x: x.split('-')[1].strip())
datos_movimientos = datos_movimientos.sort_values(by=['Sector_Pasillo', 'Columna'], ascending=[True, False])




with st.expander("Análisis de Inventario"):


    df_reposiciones = datos['Artículo'].value_counts().reset_index()
    df_reposiciones.columns = ['Artículo', 'cantidad_posiciones']
    df_reposiciones = pd.merge(df_reposiciones, datos, on='Artículo', how='left')
    df_reposiciones = df_reposiciones.drop_duplicates(subset=["Artículo"])
    df_reposiciones["diferencia"] = df_reposiciones['cantidad_repo'] - df_reposiciones["cantidad_posiciones"]
    df_reposiciones = df_reposiciones.sort_values(by=['diferencia'], ascending=[False])

    pd.set_option('display.max_columns', None)
    st.write(df_reposiciones[["Artículo", "cantidad_posiciones", "cantidad_repo","diferencia"]][df_reposiciones['cantidad_repo'] > 1])

    # Longitud objetivo de la lista
    n = len(reemplazos)

    # Crear lista de códigos de artículos según las diferencias
    lista_codigos = []

    # Iterar sobre cada fila en el DataFrame
    for _, fila in df_reposiciones.iterrows():
        # Obtener el código de artículo
        codigo_articulo = fila['Artículo']
        # Obtener la diferencia de la fila
        diferencia = fila['diferencia']

        # Repetir el código de artículo tantas veces como la diferencia
        repeticiones = int(diferencia)
        lista_codigos.extend([codigo_articulo] * repeticiones)


    # Si la lista es más larga que la longitud objetivo, recortar la lista
    if len(lista_codigos) > n:
        lista_codigos = lista_codigos[:n]

    print(lista_codigos)

    reemplazo_dic = dict(zip(reemplazos, lista_codigos))

    print(reemplazo_dic)

    datos['Artículo'].replace(reemplazo_dic, inplace=True)


    # Calcular la cantidad de veces que aparece cada artículo en 'datos_movimientos'


    # stock_articulos = datos[["Ar"]]
    zero_movimientos = datos[["Artículo", "ART.DESC.", "cantidad_movimientos", "STOCK","Sector_Pasillo_Num","Columna"]][(datos['cantidad_movimientos'] <= 1) & (datos['Artículo'] != 0)]
    # zero_movmientos = pd.merge(zero_movmientos, stock_articulos, on='Artículo', how='left')
    st.write(zero_movimientos)
    print(len(datos[["Artículo","ART.DESC.","cantidad_movimientos"]][datos['cantidad_movimientos'] <= 1]))

    vacias = len(datos[datos["Artículo"]==0])

    print(f"Cantidad de posiciones vacías: {vacias}")




    alturas = datos.groupby("Alto")["Artículo"].count().reset_index()


    datos['cat_altura'], limites = pd.qcut(datos['Alto'], 4, labels=['Short', 'Medium', 'Tall',"Super Tall"], retbins=True)
    print("Límites de los *bins* alturas:", limites)


    plt.hist(datos['Alto'], bins="auto", color='blue', alpha=0.7) # Puedes ajustar el número de bins según tus datos
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de la columna "valores"')
    plt.grid(True)
    plt.show()





with st.sidebar:
# Crear un widget de selección para mostrar o no las geometrías
    mostrar_fragiles = st.checkbox('Frágiles',value=True)
    limites_origin = st.checkbox('Limites Origin',value=True)
    limites_origines = eval(st.text_input("Lista Limites", value="[0,8,15,26]"))
    lista_valores_fragiles = st.text_input("Ingrese Codigos de Artículos frágiles", value="0,3601,3600,3651,3650,3410,3411,3603,3481")
    lista_valores_fragiles  = [int(elemento) for elemento in lista_valores_fragiles.split(',')] if lista_valores_fragiles else []
    reemplazos_1 = st.text_input("Ingrese Codigos de Artículos a reemplazar")
    limite_densidad_input = st.number_input("Limite de densidad", min_value=0.0, value=200000000.0, step=5.0)
    primeros_1 = st.text_input("Ingrese Codigos de Artículos que van al incio del recorrido")
    primeros_1 = [int(elemento) for elemento in primeros_1.split(',')] if primeros_1 else []
    direccion_livianos_1 = st.selectbox("Dirección livianos",["vertical","horizontal"],index=1)
    # Obtener las columnas del DataFrame
    columnas = datos.columns.tolist()
    mostrar_columna = st.selectbox('Columna:', columnas, index=columnas.index("cantidad_movimientos"))
    assig_sort_value_limite_1 = st.selectbox('Orden para limite 1:', columnas, index=columnas.index("cantidad_movimientos"))
    assig_sort_value_limite_2 = st.selectbox('Orden para limite 2:', columnas, index=columnas.index("cantidad_movimientos"))


# Definir una función para asignar un valor de ordenamiento dependiendo del valor de "Limite"
def assign_sort_value(row):
    if row['Limite'] == 1:
        return row[assig_sort_value_limite_1]
    elif row['Limite'] == 0:
        return row[assig_sort_value_limite_2]
    else:
        return 0  # Otro valor de "Limite"

if mostrar_fragiles:
    datos['FRAGIL'] = datos['Artículo'].apply(lambda x: 1 if x in lista_valores_fragiles else 0)
# datos['FRAGIL'] = datos['ART.DESC.'].apply(lambda x: 1 if isinstance(x, str) and any(palabra in x for palabra in ["CEREAL"]) else 0)
else:
    datos["FRAGIL"] = 0

    
datos['Limite'] = np.where(datos['Densidad'] >= limite_densidad_input, 1, 0)

#Combinar tablas


#Crear nueva columna
datos ["Cantidad/Lineas"] = (datos["Cantidad"]/datos["cantidad_movimientos"]).round(1)

col1,spacer,col2 = st.columns([2,0.1, 1])  # Col1 será más ancha que Col2

with col1:
    

    mapa_seleccionado = st.selectbox('', ["Mapa Actual","Recorridos Picking","Slotting"],index=1)

    if mapa_seleccionado =="Mapa Actual":

        # Crear una máscara para los artículos con valor 0 en la columna "Artículo"
        mascara_articulos_cero = datos['Artículo'] != 0

        datos["Sector_Pasillo_Num_NEW"] = datos["Sector_Pasillo_Num"]
        datos["Columna_NEW"] = datos["Columna"]

        st.subheader('Editar Posiciones de los Artículos')
        datos = st.data_editor(datos, num_rows="dynamic")

        # Crear el gráfico de dispersión con rectángulos usando Plotly
        fig = go.Figure()

        # Agregar puntos con artículos diferentes de 0
        fig.add_trace(go.Scatter(
            x=datos[mascara_articulos_cero]['Sector_Pasillo_Num_NEW'],
            y=datos[mascara_articulos_cero]['Columna_NEW'],
            mode='markers',
            marker=dict(
                symbol='square',
                size=15,
                color=datos[mascara_articulos_cero][mostrar_columna],
                colorscale='RdBu',
                showscale=True,
                line=dict(width=1, color='Black')
            ),
            text=datos[mascara_articulos_cero][mostrar_columna].astype(str) + '<br>' + datos[mascara_articulos_cero]['ART.DESC.'] + '<br> Peso: ' + datos[mascara_articulos_cero]["Peso"].astype(str)+ '<br> Altura: ' + datos[mascara_articulos_cero]["Alto"].astype(str) ,
            hoverinfo='text'
        ))

        # Agregar puntos con artículos igual a 0
        fig.add_trace(go.Scatter(
            x=datos[~mascara_articulos_cero]['Sector_Pasillo_Num_NEW'],
            y=datos[~mascara_articulos_cero]['Columna_NEW'],
            mode='markers',
            marker=dict(
                symbol='square',
                size=15,
                color='green',
                line=dict(width=1, color='Black')
            ),
            name='Artículo = 0'
        ))

        # Personalizar el gráfico
        fig.update_layout(
            title=f'Distribución en el Almacén ({len(datos)} Posiciones)',
            xaxis_title='Pasillo',
            yaxis_title='Columna',
            yaxis_autorange='reversed',  # Invertir el eje y para mostrar de arriba hacia abajo
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis=dict(tickmode='linear', dtick=1,showgrid=False),
            showlegend=False,
            width=800,
            height=600
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)

        reordenamiento_actual = datos[["Artículo","ART.DESC.","Sector_Pasillo_Num","Columna","Sector_Pasillo_Num_NEW","Columna_NEW"]]

        mapeo_numeros_a_letras = {valor: clave for clave, valor in mapeo_letras_a_numeros.items()}

        def convertir_numeros_a_letras(numero):
            if numero <= 26:
                if numero in mapeo_numeros_a_letras:
                    return mapeo_numeros_a_letras[numero]
                else:
                    return str(numero)  # Si el número no tiene un equivalente en el mapeo, devolver el número como cadena
            else:
                parte1 = numero // 26  # Obtener la primera parte del número
                parte2 = numero % 26   # Obtener la segunda parte del número
                letra1 = mapeo_numeros_a_letras.get(parte1, "")  # Buscar la primera parte en el diccionario inverso
                letra2 = mapeo_numeros_a_letras.get(parte2, "")  # Buscar la segunda parte en el diccionario inverso
                return letra1 + letra2  # Concatenar las letras encontradas

        reordenamiento_actual['Sector_Pasillo_Num'] = reordenamiento_actual['Sector_Pasillo_Num'].apply(convertir_numeros_a_letras)
        reordenamiento_actual['Sector_Pasillo_Num_NEW'] = reordenamiento_actual['Sector_Pasillo_Num_NEW'].apply(convertir_numeros_a_letras)

        # display(reordenamiento_actual)

        nombres_nuevos = {'Sector_Pasillo_Num': 'Pasillo_Origen', 'Columna': 'Columna_Origen','Sector_Pasillo_Num_NEW': 'Pasillo_destino', 'Columna_NEW': 'Columna_destino',  }
        reordenamiento_actual = reordenamiento_actual.rename(columns=nombres_nuevos)
        st.write(reordenamiento_actual)
    # Calcular el rango de valores de "Peso"
    peso_minimo = datos['Peso'].min()
    peso_maximo = datos['Peso'].max()
    rango_peso = peso_maximo - peso_minimo

    # Determinar los límites de cada categoría de peso
    limite_pesado = peso_minimo + rango_peso * 0.6
    limite_liviano = peso_minimo + rango_peso * 0.3

    # Función para asignar la categoría de peso a cada artículo
    def asignar_cat_peso(peso):
        if peso >= limite_pesado:
            return "Muy pesado"
        elif peso >= limite_liviano:
            return "Pesado"
        else:
            return "Liviano"

    print("Limite Pesado':", f"({limite_pesado})")
    print("Limite Liviano':", f"({limite_liviano})")


    datos['cat_peso'] = datos['Peso'].apply(asignar_cat_peso)

    datos['cat_peso'], limites = pd.cut(datos['Peso'], bins=3, labels=['Liviano', 'Pesado', 'Muy pesado'], retbins=True)
    print("Límites de los *bins*:", limites)


    if limites_origin:
        datos['cat_peso'], limites = pd.cut(datos['Peso'], bins=limites_origines, labels=['Liviano', 'Pesado', 'Muy pesado'], retbins=True)

    # Asignar la categoría de peso a cada artículo

    # Crear un diccionario de mapeo para asignar valores numéricos a cada valor único en la columna
    mapeo_valores = {'Muy pesado': 3, 'Pesado': 2, 'Liviano': 1}  # Agrega todos los valores únicos y sus correspondientes valores numéricos

    # Aplicar el mapeo a la columna 'columna_a_convertir'

    #ARTILUGIOS POSICIONES VACIAS##

    datos['cat_peso_num'] = datos['cat_peso'].map(mapeo_valores)
    datos['cat_peso_num'] = datos['cat_peso_num'].where(datos['FRAGIL'] != 1, 1)
    datos['Peso'] = datos['Peso'].where(datos['Artículo'] != 0, 200000000000000)
    # datos['Densidad'] = datos['Peso'].where(datos['Artículo'] != 0, 200000000000000)
    # datos['Densidad'] = datos.apply(lambda row: 20000000000 if row['Artículo'] in lista_valores else row['Densidad'], axis=1)

    if primeros_1:
        datos.loc[datos['Artículo'].isin(primeros_1), 'cat_peso_num'] = 3

    # Calcular el rango de valores de "cantidad_movimientos"
    movimientos_minimo = datos['cantidad_movimientos'].min()
    movimientos_maximo = datos['cantidad_movimientos'].max()
    rango_movimientos = movimientos_maximo - movimientos_minimo

    # Determinar los límites de cada categoría de rotación
    limite_media_rotacion = movimientos_minimo + rango_movimientos * 0.6
    limite_baja_rotacion = movimientos_minimo + rango_movimientos * 0.3

    # Función para asignar la categoría de rotación a cada artículo
    def asignar_cat_rotacion(cantidad_movimientos):
        if cantidad_movimientos >= limite_media_rotacion:
            return "Alta Rotación"
        elif cantidad_movimientos >= limite_baja_rotacion:
            return "Media Rotación"
        else:
            return "Baja Rotación"

    print("Limite Media rotacion':", f"({limite_media_rotacion})")
    print("Limite Baja rotacion':", f"({limite_baja_rotacion})")



    # Asignar la categoría de rotación a cada artículo
    datos['cat_rotacion'] = datos['cantidad_movimientos'].apply(asignar_cat_rotacion)

    # Calcular el rango de valores de "cantidad_movimientos"
    densidad_minimo = datos['Densidad'].min()
    densidad_maximo = datos['Densidad'].max()
    rango_densidad = densidad_maximo - densidad_minimo

    # Determinar los límites de cada categoría de rotación
    limite_media_densidad = densidad_minimo + rango_densidad * 0.68
    limite_baja_densidad = densidad_minimo + rango_densidad * 0.3

    # Función para asignar la categoría de rotación a cada artículo
    def asignar_cat_densidad(densidad):
        if densidad >= limite_media_densidad:
            return "Alta Densidad"
        elif densidad >= limite_baja_densidad:
            return "Media Densidad"
        else:
            return "Baja Densidad"

    # Asignar la categoría de rotación a cada artículo
    datos['cat_densidad'] = datos['Densidad'].apply(asignar_cat_densidad)

    cantidad_posiciones = datos['ART.DESC.'].value_counts().reset_index()
    # display(datos_articulos)
    # display (datos)
    # display(datos_movimientos)

    if mapa_seleccionado == "Slotting":

            # Crear un diccionario para mapear letras a números basados en su orden alfabético
        letras = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        mapeo_letras_a_numeros = {letra: i + 1 for i, letra in enumerate(letras)}


        import pandas as pd
        import matplotlib.pyplot as plt
        from collections import defaultdict

        # Supongamos que ya tienes tus datos y los has limpiado y ordenado según lo necesario

        # Ordenar los datos por "cat_peso_num" de forma ascendente
        datos = datos.sort_values(by='cat_peso_num',ascending=False)

        # Crear una columna temporal para el valor de ordenamiento
        datos['Sort_Value'] = datos.apply(assign_sort_value, axis=1)

        # Filtrar los datos en dos DataFrames separados
        datos_cat_peso_1 = datos[datos['cat_peso_num'] == 3]
        datos_cat_peso_2 = datos[datos['cat_peso_num'].isin([1, 2])]
        # datos_fragil = datos[datos['cat_peso_num'].isin([1, 2]) & datos['FRAGIL']== 1 ]



        datos_cat_peso_1 = datos_cat_peso_1.sort_values(by='Peso', ascending=False)
        datos_cat_peso_2 = datos_cat_peso_2.sort_values(by=['cat_peso_num',"Limite", 'Sort_Value'], ascending=False)


        datos = pd.concat([datos_cat_peso_1, datos_cat_peso_2])


        datos.drop(columns=['Sort_Value'], inplace=True)


        almacen = almacen.sort_values(by=['Sector_Pasillo_Num',"Columna"],ascending=False)
        # display(almacen)
        almacen = almacen.rename(columns={'Sector_Pasillo_Num': 'Sector_Pasillo_Num_NEW', 'Columna': 'Columna_NEW'})


        dirección_fragiles = datos[datos["FRAGIL"]==1]
        # dirección_fragiles = dirección_fragiles.sort_values(by='STOCK',ascending=False)
        dirección_fragiles = dirección_fragiles.sort_values(by='Artículo', key=lambda x: x.map({v: i for i, v in enumerate(lista_valores_fragiles)}))
        st.write(dirección_fragiles)

        # display(dirección_fragiles)

        no_fragiles = datos[datos["FRAGIL"]==0]
        dirección_1 = no_fragiles[(no_fragiles['cat_peso_num'] == 1) & (no_fragiles['Limite'] == 0) ] #Direccion para livianos no densos (rotacion)
        dirección_2 = no_fragiles.loc[~((no_fragiles['cat_peso_num'] == 1) & (no_fragiles['Limite'] == 0) )] #Direccion para pesados y densos

        # display(dirección_1)
        # display(dirección_2)


        print(datos.shape[0])
        print(dirección_1.shape[0])
        print(dirección_2.shape[0])
        print(dirección_fragiles.shape[0])

        # Obtener el número de filas del DataFrame original
        num_filas = dirección_2.shape[0]
        num_filas_fragil = dirección_fragiles.shape[0]

        print(num_filas)
        print(num_filas_fragil)


        almacen_direccion_2 = almacen.iloc[:num_filas]
        indices_filas_seleccionadas = almacen_direccion_2.index

        print(indices_filas_seleccionadas)

        almacen_direccion_fragiles = almacen.drop(indices_filas_seleccionadas)
        almacen_direccion_fragiles = almacen_direccion_fragiles.sort_values(by=['Sector_Pasillo_Num_NEW', 'Columna_NEW'], ascending=[True, False])
        almacen_direccion_fragiles = almacen_direccion_fragiles.iloc[:num_filas_fragil]
        indices_filas_seleccionadas_fragil = almacen_direccion_fragiles.index



        almacen_direccion_1 = almacen.drop(indices_filas_seleccionadas)
        almacen_direccion_1 = almacen_direccion_1.drop(indices_filas_seleccionadas_fragil)

        if direccion_livianos_1 == "vertical":
            almacen_direccion_1 = almacen_direccion_1.sort_values(by=["Sector_Pasillo_Num_NEW"],ascending=False)
        else:
            almacen_direccion_1 = almacen_direccion_1.sort_values(by=["Columna_NEW"],ascending=False)


        almacen_direccion_1.reset_index(drop=True, inplace=True)
        dirección_1.reset_index(drop=True, inplace=True)
        dirección_1 = pd.concat([dirección_1, almacen_direccion_1], axis=1)

        # display(almacen_direccion_1)


        almacen_direccion_fragiles.reset_index(drop=True, inplace=True)
        dirección_fragiles.reset_index(drop=True, inplace=True)
        dirección_fragiles = pd.concat([dirección_fragiles, almacen_direccion_fragiles], axis=1)

        # display(almacen_direccion_fragiles)

        almacen_direccion_2.reset_index(drop=True, inplace=True)
        dirección_2.reset_index(drop=True, inplace=True)
        dirección_2 = pd.concat([dirección_2, almacen_direccion_2], axis=1)

        # display(almacen_direccion_2)

        dirección_2 = dirección_2.reset_index(drop=True)
        dirección_fragiles = dirección_fragiles.reset_index(drop=True)
        dirección_1 = dirección_1.reset_index(drop=True)


        slotting = pd.concat([dirección_2, dirección_1,dirección_fragiles], ignore_index=True)
        slotting['cantidad_movimientos'] = slotting['cantidad_movimientos'].fillna(0).round(0).astype(int)
        slotting['Artículo'] = slotting['Artículo'].fillna(0).round(0).astype(int)



        cantidad_pasillos = slotting['Sector_Pasillo_Num_NEW'].nunique()


        # Crear una máscara para los artículos con valor 0 en la columna "Artículo"
        mascara_articulos_cero = slotting['Artículo'] != 0

        st.subheader('Editar Posiciones de los Artículos')
        slotting = st.data_editor(slotting, num_rows="dynamic")

        # Crear el gráfico de dispersión con rectángulos usando Plotly
        fig_2 = go.Figure()

        # Agregar puntos con artículos diferentes de 0
        fig_2.add_trace(go.Scatter(
            x=slotting[mascara_articulos_cero]['Sector_Pasillo_Num_NEW'],
            y=slotting[mascara_articulos_cero]['Columna_NEW'],
            mode='markers',
            marker=dict(
                symbol='square',
                size=15,
                color=slotting[mascara_articulos_cero][mostrar_columna],
                colorscale='RdBu',
                showscale=True,
                line=dict(width=1, color='Black')
            ),
            text=slotting[mascara_articulos_cero]["Artículo"].astype(str) + "</br>" + slotting[mascara_articulos_cero][mostrar_columna].astype(str) + '<br>' + slotting[mascara_articulos_cero]['ART.DESC.'] + '<br> Peso: ' + slotting[mascara_articulos_cero]["Peso"].astype(str)+ '<br> Alto: ' + slotting[mascara_articulos_cero]["Alto"].astype(str),
            hoverinfo='text'
        ))

        # Agregar puntos con artículos igual a 0
        fig_2.add_trace(go.Scatter(
            x=slotting[~mascara_articulos_cero]['Sector_Pasillo_Num_NEW'],
            y=slotting[~mascara_articulos_cero]['Columna_NEW'],
            mode='markers',
            marker=dict(
                symbol='square',
                size=15,
                color='green',
                line=dict(width=1, color='Black')
            ),
            name='Artículo = 0'
        ))

        # Personalizar el gráfico
        fig_2.update_layout(
            title=f'Distribución en el Almacén ({len(slotting)} Posiciones)',
            xaxis_title='Pasillo',
            yaxis_title='Columna',
            yaxis_autorange='reversed',  # Invertir el eje y para mostrar de arriba hacia abajo
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis=dict(tickmode='linear', dtick=1,showgrid=False),
            showlegend=False,
            width=800,
            height=600
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig_2)

        # # Sección para capturar la interacción y mostrar información
        # selected_data = st.empty()

        # # Función para actualizar la selección
        # def update_selection(trace, points, state):
        #     indices = points.point_inds
        #     if indices:
        #         selected_point = slotting.iloc[indices[0]]
        #         selected_data.write(f"Seleccionaste:\n"
        #                             f"Pasillo: {selected_point['Sector_Pasillo_Num_NEW']}\n"
        #                             f"Columna: {selected_point['Columna_NEW']}\n"
        #                             f"Descripción: {selected_point['ART.DESC.']}\n"
        #                             f"Peso: {selected_point['Peso']}\n"
        #                             f"Alto: {selected_point['Alto']}")

        # # Añadir el evento de clic al gráfico
        # fig_2.data[0].on_click(update_selection)

        # st.plotly_chart(fig_2)


        # datos_viejos = datos[["Artículos","Sector_Pasillo_Num","Columna"]]
        # slotting = pd.merge(slotting, datos_viejos, on='Artículo', how='left')
        # display(slotting[["Artículo","ART.DESC.","cantidad_movimientos","Peso","Densidad","bultos/movimientos","Sector_Pasillo_Num","Columna","Sector_Pasillo_Num_NEW","Columna_NEW"]])

        pd.set_option('display.max_columns', None)
        # display(slotting[["Artículo","ART.DESC.","Sector_Pasillo_Num","Columna","Sector_Pasillo_Num_NEW","Columna_NEW"]])

        reordenamiento = slotting[["Artículo","ART.DESC.","Sector_Pasillo_Num","Columna","Sector_Pasillo_Num_NEW","Columna_NEW"]]

        mapeo_numeros_a_letras = {valor: clave for clave, valor in mapeo_letras_a_numeros.items()}

        def convertir_numeros_a_letras(numero):
            if numero <= 26:
                if numero in mapeo_numeros_a_letras:
                    return mapeo_numeros_a_letras[numero]
                else:
                    return str(numero)  # Si el número no tiene un equivalente en el mapeo, devolver el número como cadena
            else:
                parte1 = numero // 26  # Obtener la primera parte del número
                parte2 = numero % 26   # Obtener la segunda parte del número
                letra1 = mapeo_numeros_a_letras.get(parte1, "")  # Buscar la primera parte en el diccionario inverso
                letra2 = mapeo_numeros_a_letras.get(parte2, "")  # Buscar la segunda parte en el diccionario inverso
                return letra1 + letra2  # Concatenar las letras encontradas

        reordenamiento['Sector_Pasillo_Num'] = reordenamiento['Sector_Pasillo_Num'].apply(convertir_numeros_a_letras)
        reordenamiento['Sector_Pasillo_Num_NEW'] = reordenamiento['Sector_Pasillo_Num_NEW'].apply(convertir_numeros_a_letras)

        # display(reordenamiento)

        nombres_nuevos = {'Sector_Pasillo_Num': 'Pasillo_Origen', 'Columna': 'Columna_Origen','Sector_Pasillo_Num_NEW': 'Pasillo_destino', 'Columna_NEW': 'Columna_destino',  }
        reordenamiento = reordenamiento.rename(columns=nombres_nuevos)
        st.write(reordenamiento)
        # reordenamiento.to_excel("/content/drive/MyDrive/iFlow/VKM - CDR/Milkaut/Reordenamiento/final.xlsx", index=False)


        inicial = datos['ART.DESC.'].value_counts().reset_index()
        # display(inicial)
        final = slotting['ART.DESC.'].value_counts().reset_index()
        final = cantidad_posiciones.merge(final,on='ART.DESC.', how='outer')
        # display(final)

    if mapa_seleccionado == "Recorridos Picking":

        

        # datos = pd.read_csv(r'E:\Mi unidad\iFlow\VKM - CDR\Milkaut\Posiciones\Sector 1\Mayo\Posiciones_2024051494128.csv',encoding='ISO-8859-1', sep=';', skiprows=2,usecols=range(0, 14))
        # datos_movimientos= pd.read_csv(r'E:\Mi unidad\iFlow\VKM - CDR\Milkaut\Movimientos\Abril\movimientos_abril_milkaut.csv',encoding='ISO-8859-1', sep=';')
        # datos_articulos = pd.read_csv(r'E:\Mi unidad\iFlow\VKM - CDR\Milkaut\Maestro Articulos\Mayo\articulos_sector_milkaut.csv',encoding='ISO-8859-1', sep=';')

        # # Extraer las partes relevantes de la posición

        # # datos_movimientos = datos_movimientos[datos_movimientos["Sector O"]=="1"]


        # datos = datos[~datos['POSICION'].str.contains('FIC|FF')]
        # datos = datos.rename(columns={'ARTICULO': 'Artículo'})
        # datos['Artículo'] = datos['Artículo'].where(datos['FP']!= 0, 0)
        # # datos = datos [datos["FP"]==1]
        # # print(datos["Artículo"].unique())
        # datos['Artículo'] = datos['Artículo'].astype('int64')
        # datos['Sector_Pasillo'] = datos['POSICION'].apply(lambda x: ''.join(c for c in x.split('-')[0].strip() if c.isalpha()))
        # datos['Columna'] = datos['POSICION'].apply(lambda x: x.split('-')[1].strip())
        # datos = datos.sort_values(by=['Sector_Pasillo', 'Columna'], ascending=[True, False])
        # letras = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # mapeo_letras_a_numeros = {letra: i + 1 for i, letra in enumerate(letras)}
        # datos['Sector_Pasillo_Num'] = datos['Sector_Pasillo'].apply(lambda x: sum(mapeo_letras_a_numeros[letra] * (26 ** i) for i, letra in enumerate(reversed(x))))
        # datos['Sector_Pasillo_Num'] = datos['Sector_Pasillo_Num'].astype(int)
        # datos['Columna'] = datos['Columna'].astype(int)

        # datos_articulos = datos_articulos[~datos_articulos['Codigo Largo'].isin(["Parlog","Pchep","Pplastico","Pavon2m","Pdescartable","testsistemas","VARIOS"])]
        # datos_articulos = datos_articulos.dropna(subset=['Codigo Largo'])
        # datos_articulos = datos_articulos.rename(columns={'Codigo Largo': 'Artículo'})
        # datos_articulos['Artículo'] = datos_articulos['Artículo'].astype('int64')



        # datos_movimientos = pd.merge(datos_movimientos, datos_articulos[['Artículo', 'Volumen']], on='Artículo', how='left')
        # datos_movimientos["Volumen"] = (datos_movimientos["Volumen"]*datos_movimientos["Cantidad"])/100000
        # datos_movimientos['Gramos'] = datos_movimientos['Gramos'].str.lstrip('=')
        # datos_movimientos['Gramos'] = datos_movimientos['Gramos'].apply(lambda x: float(eval(x)))
        # datos_movimientos['Inicio'] = pd.to_datetime(datos_movimientos['Inicio'], format='%d/%m/%y %H:%M:%S')
        # datos_movimientos['Fin'] = pd.to_datetime(datos_movimientos['Fin'], format='%d/%m/%y %H:%M:%S')
        # datos_movimientos["Duracion"] = datos_movimientos["Fin"] - datos_movimientos["Inicio"]
        # datos_movimientos['Duracion_horas'] = datos_movimientos["Duracion"].dt.total_seconds()/3600


        # # display(df_picking)


        # datos_movimientos = datos_movimientos.dropna(subset=['Posición O'])
        # datos_movimientos = datos_movimientos.dropna(subset=['Posición D'])
        # datos_movimientos['Sector_Pasillo'] = datos_movimientos['Posición O'].apply(lambda x: ''.join(c for c in x.split('-')[0].strip() if c.isalpha()))
        # datos_movimientos['Columna'] = datos_movimientos['Posición O'].apply(lambda x: x.split('-')[1].strip())
        # datos_movimientos['Sector_Pasillo_D'] = datos_movimientos['Posición D'].apply(lambda x: ''.join(c for c in x.split('-')[0].strip() if c.isalpha()))
        # datos_movimientos['Columna_D'] = datos_movimientos['Posición D'].apply(lambda x: x.split('-')[1].strip())
        # datos_movimientos = datos_movimientos.sort_values(by=['Sector_Pasillo', 'Columna'], ascending=[True, False])
        # # Convertir las letras de cada valor en "Sector_Pasillo" a números y combinarlos
        # datos_movimientos['Sector_Pasillo_Num'] = datos_movimientos['Sector_Pasillo'].apply(lambda x: sum(mapeo_letras_a_numeros[letra] * (26 ** i) for i, letra in enumerate(reversed(x))))
        # datos_movimientos['Sector_Pasillo_Num'] = datos_movimientos['Sector_Pasillo_Num'].astype(int)
        # datos_movimientos['Columna'] = datos_movimientos['Columna'].astype(int)
        # datos_movimientos['Sector_Pasillo_Num_D'] = datos_movimientos['Sector_Pasillo_D'].apply(lambda x: sum(mapeo_letras_a_numeros[letra] * (26 ** i) for i, letra in enumerate(reversed(x))))
        # datos_movimientos['Sector_Pasillo_Num_D'] = datos_movimientos['Sector_Pasillo_Num_D'].astype(int)
        # datos_movimientos['Columna_D'] = datos_movimientos['Columna_D'].astype(int)



        # Desrcipión palet

        # group_by_pallet = datos_movimientos.groupby(["Paleta"])["Cantidad"].sum().reset_index()
        # promedio = group_by_pallet["Cantidad"].mean()
        # # print(f"Promedio Bultos por Palet {promedio} blt/paleta")
        # group_by_pallet = group_by_pallet [group_by_pallet["Paleta"]!=0]
        # group_by_pallet = group_by_pallet [group_by_pallet["Paleta"]>2200000]
        # # # display(group_by_pallet)
        # # Crear el box plot
        # plt.boxplot(group_by_pallet['Cantidad'])

        # # Agregar título y etiquetas
        # plt.title('Box Plot de Datos')
        # plt.xlabel('Datos')
        # plt.ylabel('Valores')

        # # Mostrar el box plot
        # plt.show()

        # datos_movimientos['Columna'] = datos_movimientos['Columna'].astype(int)
        # Productividad
        st.write(datos_movimientos)
        df_picking = datos_movimientos[datos_movimientos["Tipo"]=="PI"]
        recorrida =df_picking.groupby(["Paleta", "Sector_Pasillo_Num"])["Columna"].agg(lambda x: x.max() - x.min()).reset_index()
        recorrida = recorrida.groupby("Paleta")["Columna"].sum().reset_index()
        articulos_group= df_picking.groupby("Paleta")["Artículo"].count().reset_index()
        recorrida = pd.merge(recorrida, articulos_group, on='Paleta', how='left')
        recorrida["Posiciones/Articulos"] = recorrida["Columna"]/recorrida["Artículo"]
        desplazamiento_primedio = round(recorrida["Posiciones/Articulos"].mean(),2)
        print(f"Desplazamiento promedio de {desplazamiento_primedio} pos/articulo")
        # Crear el box plot
        plt.boxplot(recorrida["Posiciones/Articulos"])

        # Agregar título y etiquetas
        plt.title('Box Plot de Datos')
        plt.xlabel('Datos')
        plt.ylabel('Posicones')

        # Mostrar el box plot
        plt.show()
        # display(articulos_group)
        # display(recorrida)

        productividad = df_picking["Cantidad"].sum()/df_picking["Duracion_horas"].sum()
        print(f"Productivdad mensual de {productividad} blt/hh")


        # Obtener los valores únicos de la columna "Sector_Pasillo_Num" del segundo DataFrame
        valores_unicos = datos['Sector_Pasillo_Num'].unique()
        datos_movimientos = datos_movimientos[datos_movimientos['Sector_Pasillo_Num_D'].isin(valores_unicos)]

        st.button("Reset", type="primary")
        valor_paleta = datos_movimientos['Paleta'].sample(1).iloc[0]  # Elegir un valor al azar de la columna 'Paleta'
        # valor_paleta = 2760506

        # Filtrar los datos para el valor específico de 'Paleta'
        datos_paleta = datos_movimientos[datos_movimientos["Tipo"]=="PI"]
        datos_paleta = datos_movimientos[datos_movimientos['Paleta'] == valor_paleta]
        datos_paleta = datos_paleta[datos_paleta['Sector_Pasillo_Num_D'].isin(valores_unicos)]


        articulos_paleta = datos_paleta["Artículo"].unique()
        fecha_inicial_minima = datos_paleta['Inicio'].min()
        fecha_final_maxima = datos_paleta['Fin'].max()
        operarios_paleta = datos_paleta["Operario"].unique()

        print(fecha_inicial_minima,fecha_final_maxima)

        rp = datos_movimientos[datos_movimientos["Tipo"]=="RP"]
        rp = rp[(rp['Inicio'] >= fecha_inicial_minima) & (rp['Inicio'] <= fecha_final_maxima)]
        # Filtrar el primer DataFrame para obtener solo las filas cuyos valores en la columna "Sector_Pasillo" están en los valores únicos de la lista
        rp = rp[rp['Sector_Pasillo_Num_D'].isin(valores_unicos) & rp['Artículo'].isin(articulos_paleta)]


        # display(rp)

        recorrida = datos_paleta.groupby("Sector_Pasillo_Num")["Columna"].agg(lambda x: x.max() - x.min()).reset_index()
        # display(recorrida)


        peso_total_paleta = datos_paleta['Gramos'].sum()/1000
        volumen_total_paleta = round(datos_paleta["Volumen"].sum(),1)
        cantidad_bultos_paleta = round(datos_paleta["Cantidad"].sum(),1)
        cantidad_articulos = round(datos_paleta["Artículo"].count(),1)
        tamaño_punto = datos_paleta['Cantidad']
        tipo = datos_paleta["Tipo"].unique()

        fecha_paleta = datos_paleta["Inicio"]
        print(fecha_paleta)
        tiempo_total_paleta = round(datos_paleta['Duracion_horas'].sum()*60,2)
        palet_pi = datos_paleta[datos_paleta["Tipo"]=="PI"]
        productividad_palet = round(palet_pi["Cantidad"].sum()/palet_pi["Duracion_horas"].sum(),1)

        datos_paleta['Inicio'] = pd.to_datetime(datos_paleta['Inicio'])
        datos_paleta.sort_values(by='Inicio', ascending=True, inplace=True)

        # # Crear el scatter plot
        # plt.figure(figsize=(10, 8))

        # # Ajustar los datos a los ejes x e y
        # x = datos['Sector_Pasillo_Num']
        # y = datos['Columna']

        # # Grafica los puntos para todas las posiciones
        # plt.scatter(x, y, marker='s', s=250, alpha=0.8)

        # # Ajustar el espacio entre los pasillos en el eje x
        # plt.xticks(rotation=45, ha='right')

        # # Graficar el recorrido por las posiciones para el valor específico de 'Paleta'
        # plt.scatter(datos_paleta['Sector_Pasillo_Num'], datos_paleta['Columna'],s=tamaño_punto, color='red', label=f'Recorrido para Paleta {valor_paleta}\nPeso Total: {peso_total_paleta} kilos\nTiempo Total: {tiempo_total_paleta} minutos\nVolumen Total: {volumen_total_paleta} m3\nBultos total: {cantidad_bultos_paleta}\nArticulos total: {cantidad_articulos}\nProductividad: {productividad_palet} blt/hh\nFecha: {fecha_paleta} blt/hh')
        # primer_punto = datos_paleta.iloc[0]
        # plt.scatter(primer_punto['Sector_Pasillo_Num'], primer_punto['Columna'], s=80, color='yellow', label='Primer Punto')
        # # Unir los puntos del recorrido con líneas
        # plt.plot(datos_paleta['Sector_Pasillo_Num'], datos_paleta['Columna'], color='green',alpha=0.2, linestyle='-', linewidth=2)

        # # Graficar el recorrido por las posiciones para el valor específico de 'Paleta'
        # plt.scatter(rp['Sector_Pasillo_Num'], rp['Columna'],s=rp['Cantidad'], color='blue')
        # # Unir los puntos del recorrido con líneas
        # # plt.plot(rp['Sector_Pasillo_Num'], rp['Columna'], color='blue',alpha=0.2, linestyle=':', linewidth=1)

        # # Itera a través de cada fila del DataFrame
        # for idx, row in rp.iterrows():
        #     x = [row['Sector_Pasillo_Num'], row['Sector_Pasillo_Num_D']]
        #     y = [row['Columna'], row['Columna_D']]
        #     # Grafica una línea entre los puntos (x1, y1) y (x2, y2)
        #     plt.plot(x, y,color='blue',alpha=0.2, linestyle=':', linewidth=1)  # Opcional: etiqueta cada línea


        # # Personalizar el gráfico
        # plt.title(f'Recorrido por Posiciones para Paleta {valor_paleta}')
        # plt.xlabel('Pasillo', labelpad=10)
        # plt.ylabel('Columna')
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # plt.gca().invert_yaxis()  # Invertir el eje y para mostrar de arriba hacia abajo

        # plt.xticks(range(int(x.min()), int(x.max()) + 1, 1))  # Intervalos de 1 en 1 en el eje x
        # plt.yticks(range(int(y.min()), int(y.max()) + 1, 1))
        # plt.grid(False)

        # # Mostrar el gráfico
        # plt.show()

        # Crear el scatter plot con Plotly
        fig_3 = go.Figure()

        # Texto que aparecerá en la leyenda, incluyendo los valores deseados
        leyenda_principal = (f'Recorrido para Paleta {valor_paleta}<br>'
                     f'Peso Total: {round(peso_total_paleta,2)} kilos<br>'
                     f'Tiempo Total: {tiempo_total_paleta} minutos<br>'
                     f'Fecha: {fecha_paleta.min().strftime("%Y-%m-%d %H:%M:%S")} - {fecha_paleta.max().strftime("%Y-%m-%d %H:%M:%S")} <br>'
                     f'Operario: {operarios_paleta} <br>'
                     f'Productividad: {productividad_palet} blt/hr <br>'
                    f'Tipo: {tipo} <br>')

        # Graficar todos los puntos de las posiciones
        fig_3.add_trace(go.Scatter(
            x=datos['Sector_Pasillo_Num'],
            y=datos['Columna'],
            mode='markers',
            marker=dict(symbol='square', size=10, opacity=0.8),
            name='Todas las posiciones',
            text=datos['ART.DESC.'],
            hoverinfo='text'
        ))

                # Resaltar el primer punto del recorrido
        primer_punto = datos_paleta.iloc[0]
        fig_3.add_trace(go.Scatter(
            x=[primer_punto['Sector_Pasillo_Num']],
            y=[primer_punto['Columna']],
            mode='markers',
            marker=dict(size=15, color='yellow'),
            name='Primer Punto'
        ))

        # Unir los puntos del recorrido con líneas verdes
        fig_3.add_trace(go.Scatter(
            x=datos_paleta['Sector_Pasillo_Num'],
            y=datos_paleta['Columna'],
            mode='lines',
            line=dict(color='green', width=2, dash='solid'),
            name='Recorrido'
        ))

        # Graficar el recorrido por las posiciones para el valor específico de 'Paleta'
# Graficar el recorrido por las posiciones para el valor específico de 'Paleta'
        fig_3.add_trace(go.Scatter(
            x=datos_paleta['Sector_Pasillo_Num'],
            y=datos_paleta['Columna'],
            mode='markers',
            marker=dict(size=tamaño_punto, color='red'),
            name=leyenda_principal,  # Aquí se incluye el texto en la leyenda
            text=[f'Peso Total: {peso_total_paleta} kilos<br>'
                f'Tiempo Total: {tiempo_total_paleta} minutos<br>'
                f'Volumen Total: {volumen_total_paleta} m3<br>'
                f'Bultos total: {cantidad_bultos_paleta}<br>'
                f'Articulos total: {cantidad_articulos}<br>'
                f'Productividad: {productividad_palet} blt/hh<br>']
        ))


        # Graficar puntos adicionales con tamaño proporcional a 'Cantidad'
        fig_3.add_trace(go.Scatter(
            x=rp['Sector_Pasillo_Num'],
            y=rp['Columna'],
            mode='markers',
            marker=dict(size=rp['Cantidad']*0.1, color='blue'),
            name='Reposiciones'
        ))

        # Unir los puntos adicionales con líneas
        for idx, row in rp.iterrows():
            fig_3.add_trace(go.Scatter(
                x=[row['Sector_Pasillo_Num'], row['Sector_Pasillo_Num_D']],
                y=[row['Columna'], row['Columna_D']],
                mode='lines',
                line=dict(color='blue', width=1, dash='dot'),
                showlegend=False
            ))

        # Personalizar el gráfico
        fig_3.update_layout(
            title=f'Recorrido por Posiciones para Paleta {valor_paleta}',
            xaxis_title='Pasillo',
            yaxis_title='Columna',
            # yaxis=dict(),  # Invertir el eje y para mostrar de arriba hacia abajo
            legend=dict(x=1, y=1),
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis=dict(tickmode='linear', dtick=1,autorange='reversed'),
            width=800,
            height=600
        )

        # Mostrar el gráfico con Streamlit
        st.plotly_chart(fig_3)


with col2:

    agrupado_pallet = datos_movimientos.groupby("Paleta").agg({'Kg': 'sum',"Duracion_horas":"sum"}).reset_index()
    agrupado_pallet["Duracion_minutos"] = agrupado_pallet['Duracion_horas']*60
    minutos_pallet_mean = agrupado_pallet["Duracion_horas"].mean()*60

    # fig = go.Figure()

    # fig.add_trace(go.Histogram(
    #     x=agrupado_pallet['Duracion_minutos'],
    #     nbinsx=20,  # Puedes ajustar el número de bins según lo desees
    #     marker_color='blue',  # Color de los bins
    #     opacity=0.75  # Opacidad de los bins
    # ))

    # st.plotly_chart(fig)

    suma_pesos = (datos_movimientos["Kg"]/1000).sum()
    suma_volumen = (datos_movimientos["Volumen"]).sum()
    suma_bultos = (datos_movimientos["Cantidad Blt"]).sum()
    suma_pallets = datos_movimientos['Paleta'].nunique()
    suma_horas = datos_movimientos['Duracion_horas'].sum()
    suma_tareas = datos_movimientos['Nro. Tarea'].nunique()
    suma_articulos = datos["Artículo"].nunique()
    suma_posiciones = datos["POSICION"].nunique()
    suma_posiciones_vacias =  (datos["Artículo"] == 0).sum()


    st.title("Métricas")
    st.write("A continuación se muestran las métricas relevantes:")

    choice = st.selectbox("Métrica",("Totales", "Productividad","Indices Recursos"))

    if choice == "Totales":
        col1, col2= st.columns(2)
        col1.metric("Toneladas (ton)", f"{suma_pesos:.2f}")
        col2.metric("Articulos", f"{suma_articulos:.0f}")
        col1, col2= st.columns(2)
        col1.metric("Bultos", f"{suma_bultos:.0f}")
        col2.metric("Pallets", f"{suma_pallets:.0f}")
        col1, col2= st.columns(2)
        col1.metric("Horas Hombre", f"{suma_horas:.0f}")
        col2.metric("Tareas", f"{suma_tareas:.0f}")
        col1, col2= st.columns(2)
        col1.metric("Posiciones", f"{suma_posiciones:.0f}")
        col2.metric("Posiciones Vacías", f"{suma_posiciones_vacias:.0f}")
    
    elif choice == "Productividad":
        col1, col2= st.columns(2)
        col1.metric("Bultos por Hora", f"{(suma_bultos/suma_horas):.0f}")
        col2.metric("Kg por Hora", f"{(suma_pesos/suma_horas):.0f}")
        col1, col2= st.columns(2)
        col1.metric("Pallets por Hora", f"{(suma_pallets/suma_horas):.0f}")
        col2.metric("Pallets por Viaje", f"{(suma_pallets/suma_horas):.0f}")
        col1, col2= st.columns(2)
        col1, col2= st.columns(2)
        col1.metric("Minutos por Pallet", f"{minutos_pallet_mean:.1f}")
        # col2.metric("Volumen por Parada ", f"{(suma_volumen/suma_paradas):.0f}")
        col1, col2= st.columns(2)
        # col1.metric("Bultos por Parada", f"{bultos_parada:.0f}")
        # col2.metric("Pallets por Parada", f"{(suma_pallets_volumen/suma_paradas):.0f}")

    elif choice == "Indices Recursos":
        col1, col2= st.columns(2)
        col1.metric("Peso por Pallet", f"{((suma_pesos*1000)/suma_pallets):.1f}")
        col2.metric("Peso por Bulto", f"{((suma_pesos*1000)/suma_bultos):.1f}")
        col1, col2= st.columns(2)
        col1.metric("Bultos por Pallet", f"{(suma_bultos/suma_pallets):.0f}")
        col2.metric("Volumen por Pallet", f"{(suma_volumen/suma_pallets):.0f}")
