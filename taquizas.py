import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog,Toplevel, ttk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Asegurarse de que se utiliza el backend TkAgg para mostrar gráficos en Tkinter
matplotlib.use("TkAgg")

# Función para conectar a la base de datos
def conectar_db():
    try:
        conexion = sqlite3.connect('taquizas_db.sqlite')
        return conexion
    except sqlite3.Error as e:
        messagebox.showerror("Error de conexión", f"No se pudo conectar a la base de datos: {str(e)}")
        return None


# Crear un analizador de sentimientos de VADER
analyzer = SentimentIntensityAnalyzer()

# Función para mostrar todos los registros
def mostrar_registros():
    try:
        conexion = conectar_db()
        if conexion is None:
            return  # No continuar si no se pudo conectar

        df = pd.read_sql_query("SELECT rowid, * FROM taquizas", conexion)  # Asegúrate de seleccionar rowid

        # Limpiar el Treeview antes de mostrar nuevos datos
        for item in tree.get_children():
            tree.delete(item)

        # Insertar nuevos datos con rowid
        for index, row in df.iterrows():
            tree.insert("", "end", values=[row['rowid']] + list(row[1:]))  # Usamos rowid como primer valor

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al mostrar los registros: {str(e)}")
    finally:
        if conexion is not None:
            conexion.close()


# Función para mostrar estadísticas
def mostrar_estadisticas():
    try:
        conexion = conectar_db()
        if conexion is None:
            return  # No continuar si no se pudo conectar

        df = pd.read_sql_query("SELECT * FROM taquizas", conexion)

        # Contar la cantidad de eventos por zona
        df_eventos = df['Zona'].value_counts().reset_index()
        df_eventos.columns = ['Zona', 'Cantidad de Eventos']

        # Sumar la cantidad de personas por zona
        df_personas = df.groupby('Zona')['Cantidad de personas'].sum().reset_index()
        df_personas.columns = ['Zona', 'Total de Personas']

        # Sumar el costo por zona
        df_costos = df.groupby('Zona')['Costo'].sum().reset_index()
        df_costos.columns = ['Zona', 'Suma de Costos']

        # Crear una nueva ventana para mostrar estadísticas
        ventana_estadisticas = tk.Toplevel()
        ventana_estadisticas.title("Estadísticas por zona")

        # Crear un Treeview para mostrar las estadísticas
        tabla_estadisticas = ttk.Treeview(ventana_estadisticas, columns=("Zona", "Cantidad de Eventos", "Total de Personas", "Suma de Costos"), show='headings')
        tabla_estadisticas.heading("Zona", text="Zona")
        tabla_estadisticas.heading("Cantidad de Eventos", text="Cantidad de Eventos")
        tabla_estadisticas.heading("Total de Personas", text="Total de Personas")
        tabla_estadisticas.heading("Suma de Costos", text="Suma de Costos")

        # Insertar datos en la tabla
        for i in range(len(df_eventos)):
            tabla_estadisticas.insert("", "end", values=(df_eventos.iloc[i]["Zona"], df_eventos.iloc[i]["Cantidad de Eventos"],
                                                         df_personas[df_personas["Zona"] == df_eventos.iloc[i]["Zona"]]["Total de Personas"].values[0],
                                                         df_costos[df_costos["Zona"] == df_eventos.iloc[i]["Zona"]]["Suma de Costos"].values[0]))

        tabla_estadisticas.pack(fill=tk.BOTH, expand=True)
        ventana_estadisticas.geometry("600x400")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al mostrar las estadísticas: {str(e)}")
    finally:
        if conexion is not None:
            conexion.close()

# Función para mostrar gráficos de estadísticas en una sola ventana
def mostrar_graficas_estadisticas():
    try:
        conexion = conectar_db()
        if conexion is None:
            return  # No continuar si no se pudo conectar

        df = pd.read_sql_query("SELECT * FROM taquizas", conexion)

        # Contar la cantidad de eventos por zona
        df_eventos = df['Zona'].value_counts().reset_index()
        df_eventos.columns = ['Zona', 'Cantidad de Eventos']

        # Sumar la cantidad de personas por zona
        df_personas = df.groupby('Zona')['Cantidad de personas'].sum().reset_index()
        df_personas.columns = ['Zona', 'Total de Personas']

        # Sumar el costo por zona
        df_costos = df.groupby('Zona')['Costo'].sum().reset_index()
        df_costos.columns = ['Zona', 'Suma de Costos']

        # Crear subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))  # 3 filas, 1 columna

        # Gráfico de barras para la cantidad de eventos por zona
        sns.barplot(ax=axes[0], x='Zona', y='Cantidad de Eventos', data=df_eventos, palette='viridis')
        axes[0].set_title('Cantidad de Eventos por Zona')
        axes[0].set_xlabel('Zona')
        axes[0].set_ylabel('Cantidad de Eventos')

        # Gráfico de barras para el total de personas por zona
        sns.barplot(ax=axes[1], x='Zona', y='Total de Personas', data=df_personas, palette='viridis')
        axes[1].set_title('Total de Personas por Zona')
        axes[1].set_xlabel('Zona')
        axes[1].set_ylabel('Total de Personas')

        # Gráfico de barras para la suma de costos por zona
        sns.barplot(ax=axes[2], x='Zona', y='Suma de Costos', data=df_costos, palette='viridis')
        axes[2].set_title('Suma de Costos por Zona')
        axes[2].set_xlabel('Zona')
        axes[2].set_ylabel('Suma de Costos')

        # Ajustar los espacios entre gráficos
        plt.tight_layout()

        # Mostrar todas las gráficas en una sola ventana
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al mostrar las gráficas: {str(e)}")
    finally:
        if conexion is not None:
            conexion.close()


# Función para aplicar análisis predictivo de series temporales
def analizar_series_temporales():
    try:
        conexion = conectar_db()
        if conexion is None:
            return  # No continuar si no se pudo conectar

        # Leer datos históricos de pedidos
        df = pd.read_sql_query("SELECT Fecha, `Cantidad de personas` FROM taquizas", conexion)

        # Convertir la columna Fecha a formato datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'])

        # Establecer la columna Fecha como índice
        df.set_index('Fecha', inplace=True)

        # Agrupar la cantidad de personas por mes para análisis de demanda mensual
        df_mensual = df.resample('M').sum()

        # Crear la figura y los subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))  # 2 filas, 1 columna

        # Gráfico de la demanda histórica
        axes[0].plot(df_mensual, label='Demanda histórica')
        axes[0].set_title('Demanda histórica de taquizas por mes')
        axes[0].set_xlabel('Fecha')
        axes[0].set_ylabel('Cantidad de personas')
        axes[0].legend()

        # Crear y entrenar el modelo ARIMA
        model = ARIMA(df_mensual, order=(1, 1, 1))  # Ajustar el orden según sea necesario
        model_fit = model.fit()

        # Hacer predicciones a futuro (por ejemplo, los próximos 12 meses)
        predicciones = model_fit.forecast(steps=12)

        # Gráfico de predicciones
        axes[1].plot(df_mensual, label='Demanda histórica')
        axes[1].plot(predicciones, label='Predicción de demanda', color='red')
        axes[1].set_title('Predicción de demanda de taquizas para los próximos 12 meses')
        axes[1].set_xlabel('Fecha')
        axes[1].set_ylabel('Cantidad de personas')
        axes[1].legend()

        # Ajustar los espacios entre los subplots
        plt.tight_layout()

        # Mostrar ambas gráficas en la misma ventana
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al realizar el análisis de series temporales: {str(e)}")
    finally:
        if conexion is not None:
            conexion.close()


def realizar_clustering():
    try:
        conexion = conectar_db()
        if conexion is None:
            return  # No continuar si no se pudo conectar

        # Leer datos relevantes de la base de datos
        df = pd.read_sql_query("SELECT `Nombre del solicitante`, Zona, `Cantidad de personas`, Costo FROM taquizas", conexion)

        # Crear clusters basados en la cantidad de personas
        def clasificar_cluster(cantidad):
            if 20 <= cantidad <= 70:
                return 0  # Cluster 0: Zonas con menor cantidad de personas
            elif 71 <= cantidad <= 149:
                return 1  # Cluster 1: Zonas con cantidad moderada de personas
            elif cantidad > 150:
                return 2  # Cluster 2: Zonas con taquizas grandes
            else:
                return -1  # Cluster no válido

        # Aplicar la función de clasificación
        df['Cluster'] = df['Cantidad de personas'].apply(clasificar_cluster)

        # Filtrar solo los clusters válidos
        df = df[df['Cluster'] != -1]

        # Descripción de cada cluster
        cluster_descripcion = df.groupby('Cluster').agg({
            'Zona': 'first',  # Puedes ajustar esto según necesites
            'Cantidad de personas': 'mean',
            'Costo': 'mean'
        }).rename(columns={
            'Zona': 'Zona Promedio',
            'Cantidad de personas': 'Promedio de Personas',
            'Costo': 'Costo Promedio'
        })

        # Visualización de los clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Zona', y='Cantidad de personas', hue='Cluster', data=df, palette='viridis')

        # Añadir leyenda personalizada para las zonas
        plt.title('Clustering de Clientes por Cantidad de Personas')
        plt.xlabel('Zona')
        plt.ylabel('Cantidad de Personas')
        plt.tight_layout()  # Asegura que los elementos no se superpongan
        plt.show()

        # Crear una ventana para mostrar la descripción de cada cluster
        descripcion_ventana = tk.Toplevel()
        descripcion_ventana.title("Descripción de Clusters")

        # Tabla para mostrar las características promedio de cada cluster
        tabla_descripcion = ttk.Treeview(descripcion_ventana, columns=("Cluster", "Zona Promedio", "Promedio de Personas", "Costo Promedio"), show='headings')
        tabla_descripcion.heading("Cluster", text="Cluster")
        tabla_descripcion.heading("Zona Promedio", text="Zona Promedio")
        tabla_descripcion.heading("Promedio de Personas", text="Promedio de Personas")
        tabla_descripcion.heading("Costo Promedio", text="Costo Promedio")

        # Insertar los datos agregados de cada cluster en la tabla
        for index, row in cluster_descripcion.iterrows():
            tabla_descripcion.insert("", "end", values=(index, row['Zona Promedio'], row['Promedio de Personas'], row['Costo Promedio']))
        
        tabla_descripcion.pack(fill=tk.BOTH, expand=True)

        # Tabla para mostrar a cada persona y su cluster asignado
        tabla_personas = ttk.Treeview(descripcion_ventana, columns=("Nombre del solicitante", "Zona", "Cantidad de personas", "Costo", "Cluster"), show='headings')
        tabla_personas.heading("Nombre del solicitante", text="Nombre del Solicitante")
        tabla_personas.heading("Zona", text="Zona")
        tabla_personas.heading("Cantidad de personas", text="Cantidad de Personas")
        tabla_personas.heading("Costo", text="Costo")
        tabla_personas.heading("Cluster", text="Cluster")

        # Insertar cada cliente con su cluster en la tabla
        for _, row in df.iterrows():
            tabla_personas.insert("", "end", values=(row['Nombre del solicitante'], row['Zona'], row['Cantidad de personas'], row['Costo'], row['Cluster']))

        tabla_personas.pack(fill=tk.BOTH, expand=True)
        descripcion_ventana.geometry("800x600")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al realizar el clustering: {str(e)}")
    finally:
        if conexion is not None:
            conexion.close()

# Función para realizar el análisis de sentimientos y mostrar en un Treeview
def analizar_sentimientos():
    try:
        # Conectar a la base de datos
        conexion = conectar_db()
        if conexion is None:
            return

        # Leer todos los comentarios y nombres de la base de datos
        df = pd.read_sql_query("""
            SELECT rowid, "Nombre del solicitante", "Comentario" 
            FROM taquizas 
            WHERE "Comentario" IS NOT NULL
        """, conexion)

        # Crear una nueva ventana para mostrar resultados
        ventana_resultados = Toplevel()
        ventana_resultados.title("Resultados del Análisis de Sentimientos")
        ventana_resultados.geometry("800x500")

        # Crear un Treeview para mostrar resultados
        tree_resultados = ttk.Treeview(
            ventana_resultados, 
            columns=("ID", "Nombre del Solicitante", "Comentario", "Sentimiento"), 
            show='headings'
        )
        tree_resultados.heading("ID", text="ID")
        tree_resultados.heading("Nombre del Solicitante", text="Nombre del Solicitante")
        tree_resultados.heading("Comentario", text="Comentario")
        tree_resultados.heading("Sentimiento", text="Sentimiento")
        tree_resultados.column("ID", width=50)
        tree_resultados.column("Nombre del Solicitante", width=200)
        tree_resultados.column("Comentario", width=400)
        tree_resultados.column("Sentimiento", width=100)
        tree_resultados.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Aplicar estilos de color con etiquetas (tags)
        tree_resultados.tag_configure("Positivo", background="#d4edda", foreground="#155724")  # Verde
        tree_resultados.tag_configure("Negativo", background="#f8d7da", foreground="#721c24")  # Rojo
        tree_resultados.tag_configure("Neutral", background="#fff3cd", foreground="#856404")  # Amarillo

        # Analizar sentimiento de cada comentario
        for index, row in df.iterrows():
            comentario = row['Comentario']
            nombre = row['Nombre del solicitante']
            sentiment_score = analyzer.polarity_scores(comentario)
            polaridad = sentiment_score['compound']

            # Determinar sentimiento y asignar etiqueta
            if polaridad >= 0.05:
                sentimiento = 'Positivo'
                etiqueta = "Positivo"
            elif polaridad <= -0.05:
                sentimiento = 'Negativo'
                etiqueta = "Negativo"
            else:
                sentimiento = 'Neutral'
                etiqueta = "Neutral"

            # Insertar datos en el Treeview con la etiqueta
            tree_resultados.insert("", "end", values=(row['rowid'], nombre, comentario, sentimiento), tags=(etiqueta,))

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al analizar los sentimientos: {str(e)}")
    finally:
        if conexion is not None:
            conexion.close()

# Función para agregar un nuevo registro a la base de datos
def agregar_nuevo_registro():
    def guardar_registro():
        nombre = entry_nombre.get()
        fecha = entry_fecha.get()
        horario = entry_horario.get()
        cantidad_personas = entry_cantidad_personas.get()
        direccion = entry_direccion.get()
        zona = entry_zona.get()
        tipo_evento = entry_tipo_evento.get()
        costo = entry_costo.get()
        
        try:
            conexion = conectar_db()
            if conexion is None:
                return

            cursor = conexion.cursor()
            cursor.execute(
                """
                INSERT INTO taquizas 
                ("Nombre del solicitante", "Fecha", "Horario", "Cantidad de personas", 
                 "Direccion", "Zona", "Tipo de evento", "Costo")
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (nombre, fecha, horario, cantidad_personas, direccion, zona, tipo_evento, costo)
            )
            conexion.commit()
            messagebox.showinfo("Éxito", "Registro agregado correctamente.")
            ventana_agregar.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo agregar el registro: {str(e)}")
        finally:
            if conexion:
                conexion.close()

    # Crear ventana para agregar un nuevo registro
    ventana_agregar = tk.Toplevel()
    ventana_agregar.title("Agregar Nuevo Registro")

    # Etiquetas y entradas para cada campo
    campos = ["Nombre del solicitante", "Fecha", "Horario", "Cantidad de personas", 
              "Direccion", "Zona", "Tipo de evento", "Costo"]
    entradas = {}
    for i, campo in enumerate(campos):
        label = tk.Label(ventana_agregar, text=campo)
        label.grid(row=i, column=0, padx=10, pady=5)
        entry = tk.Entry(ventana_agregar)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entradas[campo] = entry

    # Asignar entradas a variables específicas
    entry_nombre = entradas["Nombre del solicitante"]
    entry_fecha = entradas["Fecha"]
    entry_horario = entradas["Horario"]
    entry_cantidad_personas = entradas["Cantidad de personas"]
    entry_direccion = entradas["Direccion"]
    entry_zona = entradas["Zona"]
    entry_tipo_evento = entradas["Tipo de evento"]
    entry_costo = entradas["Costo"]

    # Botón para guardar el nuevo registro
    boton_guardar = tk.Button(ventana_agregar, text="Guardar", command=guardar_registro)
    boton_guardar.grid(row=len(campos), columnspan=2, pady=10)


# Crear la ventana principal
ventana_principal = tk.Tk()
ventana_principal.title("Sistema de Taquizas a Domicilio")

# Crear un Treeview para mostrar registros
tree = ttk.Treeview(
    ventana_principal,
    columns=("ID", "Nombre del solicitante", "Fecha", "Horario", "Cantidad de personas", "Direccion", "Zona", "Tipo de evento", "Costo"),
    show='headings'
)

# Definir encabezados de columna
tree.heading("ID", text="ID", command=lambda: ordenar_treeview("ID", False))  # Nuevo encabezado para ID
tree.heading("Nombre del solicitante", text="Nombre del solicitante", command=lambda: ordenar_treeview("Nombre del solicitante", False))
tree.heading("Fecha", text="Fecha", command=lambda: ordenar_treeview("Fecha", False))
tree.heading("Horario", text="Horario", command=lambda: ordenar_treeview("Horario", False))
tree.heading("Cantidad de personas", text="Cantidad de personas", command=lambda: ordenar_treeview("Cantidad de personas", False))
tree.heading("Direccion", text="Direccion", command=lambda: ordenar_treeview("Direccion", False))
tree.heading("Zona", text="Zona", command=lambda: ordenar_treeview("Zona", False))
tree.heading("Tipo de evento", text="Tipo de evento", command=lambda: ordenar_treeview("Tipo de evento", False))
tree.heading("Costo", text="Costo", command=lambda: ordenar_treeview("Costo", False))

# Ajustar el ancho de cada columna
tree.column("ID", width=50)  # Columna para ID
tree.column("Nombre del solicitante", width=200)
tree.column("Fecha", width=100)
tree.column("Horario", width=100)
tree.column("Cantidad de personas", width=150)
tree.column("Direccion", width=250)
tree.column("Zona", width=100)
tree.column("Tipo de evento", width=150)
tree.column("Costo", width=100)

# Empaquetar el Treeview
tree.pack(fill=tk.BOTH, expand=True)


def modificar_registro():
    seleccionado = tree.selection()
    if not seleccionado:
        messagebox.showwarning("Seleccionar Registro", "Debe seleccionar un registro para modificar.")
        return

    item = tree.item(seleccionado)
    valores = item['values']

    # Verificar que el rowid es el correcto
    rowid = valores[0]  # Asumiendo que el rowid está en la primera posición
    print(f"El rowid del registro seleccionado es: {rowid}")

    def guardar_modificaciones():
        nombre = entradas["Nombre del solicitante"].get()
        fecha = entradas["Fecha"].get()
        horario = entradas["Horario"].get()
        cantidad_personas = entradas["Cantidad de personas"].get()
        direccion = entradas["Direccion"].get()
        zona = entradas["Zona"].get()
        tipo_evento = entradas["Tipo de evento"].get()
        costo = entradas["Costo"].get()

        try:
            conexion = conectar_db()
            if conexion is None:
                return  # No continuar si no se pudo conectar

            cursor = conexion.cursor()

            print(f"Actualizando registro con id {rowid}: {nombre}, {fecha}, {horario}, {cantidad_personas}, {direccion}, {zona}, {tipo_evento}, {costo}")

            cursor.execute(
                """
                UPDATE taquizas
                SET "Nombre del solicitante" = ?, "Fecha" = ?, "Horario" = ?, "Cantidad de personas" = ?,
                    "Direccion" = ?, "Zona" = ?, "Tipo de evento" = ?, "Costo" = ?
                WHERE rowid = ?
                """,
                (nombre, fecha, horario, cantidad_personas, direccion, zona, tipo_evento, costo, rowid)  # rowid correcto
            )
            conexion.commit()

            if cursor.rowcount > 0:
                messagebox.showinfo("Éxito", "Registro modificado correctamente.")
            else:
                messagebox.showwarning("Advertencia", "No se realizó ninguna actualización.")
            
            ventana_modificar.destroy()
            mostrar_registros()  # Actualizar la vista de registros

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo modificar el registro: {str(e)}")
        finally:
            if conexion:
                conexion.close()

    # Ventana para modificar el registro
    ventana_modificar = tk.Toplevel()
    ventana_modificar.title("Modificar Registro")

    campos = ["Nombre del solicitante", "Fecha", "Horario", "Cantidad de personas", 
              "Direccion", "Zona", "Tipo de evento", "Costo"]
    entradas = {}

    for i, campo in enumerate(campos):
        label = tk.Label(ventana_modificar, text=campo)
        label.grid(row=i, column=0, padx=10, pady=5)
        entry = tk.Entry(ventana_modificar)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entradas[campo] = entry

    # Rellenar los campos con los valores actuales
    entradas["Nombre del solicitante"].insert(0, valores[1])  # El nombre del solicitante es el segundo valor
    entradas["Fecha"].insert(0, valores[2])
    entradas["Horario"].insert(0, valores[3])
    entradas["Cantidad de personas"].insert(0, valores[4])
    entradas["Direccion"].insert(0, valores[5])
    entradas["Zona"].insert(0, valores[6])
    entradas["Tipo de evento"].insert(0, valores[7])
    entradas["Costo"].insert(0, valores[8])

    # Botón para guardar los cambios
    boton_guardar = tk.Button(ventana_modificar, text="Guardar Cambios", command=guardar_modificaciones)
    boton_guardar.grid(row=len(campos), columnspan=2, pady=10)


def eliminar_registro():
    seleccionado = tree.selection()
    if not seleccionado:
        messagebox.showwarning("Seleccionar Registro", "Debe seleccionar un registro para eliminar.")
        return

    item = tree.item(seleccionado)
    valores = item['values']
    
    # Verificar que el rowid es válido
    rowid = valores[0]  # Supone que el ID está en la primera columna
    if not rowid:
        messagebox.showerror("Error", "No se pudo determinar el ID del registro.")
        return

    # Confirmar eliminación
    respuesta = messagebox.askyesno("Confirmar Eliminación", f"¿Está seguro de eliminar el registro con ID {rowid}?")
    if not respuesta:
        return

    try:
        conexion = conectar_db()
        if conexion is None:
            return

        cursor = conexion.cursor()
        cursor.execute("DELETE FROM taquizas WHERE rowid = ?", (rowid,))
        conexion.commit()

        if cursor.rowcount > 0:
            messagebox.showinfo("Éxito", "Registro eliminado correctamente.")
            mostrar_registros()  # Refrescar la vista
        else:
            messagebox.showwarning("Advertencia", "No se encontró el registro para eliminar.")

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo eliminar el registro: {str(e)}")
    finally:
        if conexion:
            conexion.close()

def asignar_evento():
    # Crear una ventana nueva
    ventana_asignacion = Toplevel(ventana_principal)
    ventana_asignacion.title("Asignación de colaboradores e insumos")
    ventana_asignacion.geometry("800x500")
    
    # Conectar a la base de datos
    conexion = conectar_db()
    if conexion:
        cursor = conexion.cursor()
        
        # Consulta SQL para obtener los campos deseados
        cursor.execute("""
            SELECT rowid, "Nombre del solicitante", Fecha, "Cantidad de personas", 
                   colaboradores, kg_tortilla, kg_queso, kg_tortilla_harina, 
                   kg_bistek, kg_chorizo, kg_pastor, kg_cebolla, kg_limones
            FROM taquizas
        """)
        taquizas = cursor.fetchall()

        # Crear un contenedor de marco para el Treeview y el Scroll
        frame_treeview = tk.Frame(ventana_asignacion)
        frame_treeview.pack(fill=tk.BOTH, expand=True)

        # Crear un Treeview para mostrar los datos
        tree = ttk.Treeview(frame_treeview, columns=(
            "Solicitante", "Fecha", "Cantidad", "Colaboradores", "kg_tortilla", 
            "kg_queso", "kg_tortilla_harina", "kg_bistek", "kg_chorizo", 
            "kg_pastor", "kg_cebolla", "kg_limones"), show="headings")
        
        # Definir los encabezados
        tree.heading("Solicitante", text="Nombre del Solicitante")
        tree.heading("Fecha", text="Fecha")
        tree.heading("Cantidad", text="Cantidad de Personas")
        tree.heading("Colaboradores", text="Colaboradores")
        tree.heading("kg_tortilla", text="Kg Tortilla")
        tree.heading("kg_queso", text="Kg Queso")
        tree.heading("kg_tortilla_harina", text="Kg Tortilla Harina")
        tree.heading("kg_bistek", text="Kg Bistek")
        tree.heading("kg_chorizo", text="Kg Chorizo")
        tree.heading("kg_pastor", text="Kg Pastor")
        tree.heading("kg_cebolla", text="Kg Cebolla")
        tree.heading("kg_limones", text="Kg Limones")
        
        # Agregar las columnas y configurar el Treeview
        tree.column("Solicitante", width=150)
        tree.column("Fecha", width=100)
        tree.column("Cantidad", width=100)
        tree.column("Colaboradores", width=150)
        tree.column("kg_tortilla", width=100)
        tree.column("kg_queso", width=100)
        tree.column("kg_tortilla_harina", width=120)
        tree.column("kg_bistek", width=100)
        tree.column("kg_chorizo", width=100)
        tree.column("kg_pastor", width=100)
        tree.column("kg_cebolla", width=100)
        tree.column("kg_limones", width=100)

        # Crear scrollbar horizontal
        scrollbar_x = tk.Scrollbar(frame_treeview, orient="horizontal", command=tree.xview)
        tree.config(xscrollcommand=scrollbar_x.set)
        scrollbar_x.pack(side="bottom", fill="x")

        tree.pack(fill=tk.BOTH, expand=True)

        # Agregar los registros de taquizas al Treeview
        for taquiza in taquizas:
            tree.insert("", "end", values=taquiza[1:])  # taquiza[1:] omite el rowid

        # Función que se ejecuta cuando seleccionas una taquiza
        def seleccionar_taquiza(event):
            item = tree.selection()[0]  # Obtener el ID de la taquiza seleccionada
            rowid_taquiza = taquizas[tree.index(item)][0]  # El rowid está en la primera columna de taquizas
            abrir_detalles_taquiza(rowid_taquiza)

        tree.bind("<Double-1>", seleccionar_taquiza)  # Al hacer doble clic en una taquiza, se abre la ventana de detalles


def abrir_detalles_taquiza(rowid_taquiza):
    # Crear una nueva ventana
    ventana_detalles = Toplevel(ventana_principal)
    ventana_detalles.title("Detalles de la Taquiza")
    ventana_detalles.geometry("600x400")

    # Conectar a la base de datos
    conexion = conectar_db()
    if conexion:
        cursor = conexion.cursor()

        # Obtener los detalles de la taquiza seleccionada
        cursor.execute("""
            SELECT colaboradores, kg_tortilla, kg_queso, kg_tortilla_harina,
                   kg_bistek, kg_chorizo, kg_pastor, kg_cebolla, kg_limones
            FROM taquizas
            WHERE rowid = ?
        """, (rowid_taquiza,))
        
        taquiza = cursor.fetchone()  # Trae los datos de la taquiza seleccionada

        if taquiza:
            # Reemplazar valores None por valores predeterminados
            taquiza = [valor if valor is not None else "" for valor in taquiza]

            # Crear campos de entrada para cada valor
            colaboradores_entry = tk.Entry(ventana_detalles)
            colaboradores_entry.insert(0, taquiza[0])  # Colaboradores

            kg_tortilla_entry = tk.Entry(ventana_detalles)
            kg_tortilla_entry.insert(0, taquiza[1])  # kg_tortilla
            
            kg_queso_entry = tk.Entry(ventana_detalles)
            kg_queso_entry.insert(0, taquiza[2])  # kg_queso
            
            kg_tortilla_harina_entry = tk.Entry(ventana_detalles)
            kg_tortilla_harina_entry.insert(0, taquiza[3])  # kg_tortilla_harina

            kg_bistek_entry = tk.Entry(ventana_detalles)
            kg_bistek_entry.insert(0, taquiza[4])  # kg_bistek

            kg_chorizo_entry = tk.Entry(ventana_detalles)
            kg_chorizo_entry.insert(0, taquiza[5])  # kg_chorizo

            kg_pastor_entry = tk.Entry(ventana_detalles)
            kg_pastor_entry.insert(0, taquiza[6])  # kg_pastor

            kg_cebolla_entry = tk.Entry(ventana_detalles)
            kg_cebolla_entry.insert(0, taquiza[7])  # kg_cebolla

            kg_limones_entry = tk.Entry(ventana_detalles)
            kg_limones_entry.insert(0, taquiza[8])  # kg_limones

            # Empacar los campos
            colaboradores_entry.grid(row=0, column=1, padx=10, pady=5)
            kg_tortilla_entry.grid(row=1, column=1, padx=10, pady=5)
            kg_queso_entry.grid(row=2, column=1, padx=10, pady=5)
            kg_tortilla_harina_entry.grid(row=3, column=1, padx=10, pady=5)
            kg_bistek_entry.grid(row=4, column=1, padx=10, pady=5)
            kg_chorizo_entry.grid(row=5, column=1, padx=10, pady=5)
            kg_pastor_entry.grid(row=6, column=1, padx=10, pady=5)
            kg_cebolla_entry.grid(row=7, column=1, padx=10, pady=5)
            kg_limones_entry.grid(row=8, column=1, padx=10, pady=5)

            # Etiquetas para cada campo
            tk.Label(ventana_detalles, text="Colaboradores:").grid(row=0, column=0, padx=10, pady=5)
            tk.Label(ventana_detalles, text="Kg de Tortilla:").grid(row=1, column=0, padx=10, pady=5)
            tk.Label(ventana_detalles, text="Kg de Queso:").grid(row=2, column=0, padx=10, pady=5)
            tk.Label(ventana_detalles, text="Kg Tortilla Harina:").grid(row=3, column=0, padx=10, pady=5)
            tk.Label(ventana_detalles, text="Kg Bistek:").grid(row=4, column=0, padx=10, pady=5)
            tk.Label(ventana_detalles, text="Kg Chorizo:").grid(row=5, column=0, padx=10, pady=5)
            tk.Label(ventana_detalles, text="Kg Pastor:").grid(row=6, column=0, padx=10, pady=5)
            tk.Label(ventana_detalles, text="Kg Cebolla:").grid(row=7, column=0, padx=10, pady=5)
            tk.Label(ventana_detalles, text="Kg Limones:").grid(row=8, column=0, padx=10, pady=5)

            # Botón para guardar los cambios
            guardar_button = tk.Button(ventana_detalles, text="Guardar Cambios", command=lambda: guardar_cambios(
                rowid_taquiza, colaboradores_entry, kg_tortilla_entry, kg_queso_entry, kg_tortilla_harina_entry,
                kg_bistek_entry, kg_chorizo_entry, kg_pastor_entry, kg_cebolla_entry, kg_limones_entry,
                ventana_detalles))
            guardar_button.grid(row=9, column=0, columnspan=2, pady=10)


# Función para guardar los cambios
def guardar_cambios(rowid_taquiza, colaboradores_entry, kg_tortilla_entry, kg_queso_entry,
                    kg_tortilla_harina_entry, kg_bistek_entry, kg_chorizo_entry, kg_pastor_entry,
                    kg_cebolla_entry, kg_limones_entry, ventana_detalles):
    # Obtener los nuevos valores de los campos
    colaboradores_nuevos = colaboradores_entry.get()
    kg_tortilla_nuevos = kg_tortilla_entry.get()
    kg_queso_nuevos = kg_queso_entry.get()
    kg_tortilla_harina_nuevos = kg_tortilla_harina_entry.get()
    kg_bistek_nuevos = kg_bistek_entry.get()
    kg_chorizo_nuevos = kg_chorizo_entry.get()
    kg_pastor_nuevos = kg_pastor_entry.get()
    kg_cebolla_nuevos = kg_cebolla_entry.get()
    kg_limones_nuevos = kg_limones_entry.get()

    # Mostrar los datos obtenidos en la consola para depurar
    print("Datos a guardar:")
    print(f"Colaboradores: {colaboradores_nuevos}")
    print(f"Kg Tortilla: {kg_tortilla_nuevos}")
    print(f"Kg Queso: {kg_queso_nuevos}")
    print(f"Kg Tortilla Harina: {kg_tortilla_harina_nuevos}")
    print(f"Kg Bistek: {kg_bistek_nuevos}")
    print(f"Kg Chorizo: {kg_chorizo_nuevos}")
    print(f"Kg Pastor: {kg_pastor_nuevos}")
    print(f"Kg Cebolla: {kg_cebolla_nuevos}")
    print(f"Kg Limones: {kg_limones_nuevos}")

    # Validar que los valores numéricos sean correctos
    try:
        kg_tortilla_nuevos = float(kg_tortilla_nuevos)
        kg_queso_nuevos = float(kg_queso_nuevos)
        kg_tortilla_harina_nuevos = float(kg_tortilla_harina_nuevos)
        kg_bistek_nuevos = float(kg_bistek_nuevos)
        kg_chorizo_nuevos = float(kg_chorizo_nuevos)
        kg_pastor_nuevos = float(kg_pastor_nuevos)
        kg_cebolla_nuevos = float(kg_cebolla_nuevos)
        kg_limones_nuevos = float(kg_limones_nuevos)

        # Conectar a la base de datos
        conexion = conectar_db()
        if conexion:
            cursor = conexion.cursor()
            cursor.execute("""
                UPDATE taquizas
                SET colaboradores = ?, kg_tortilla = ?, kg_queso = ?, kg_tortilla_harina = ?, kg_bistek = ?,
                    kg_chorizo = ?, kg_pastor = ?, kg_cebolla = ?, kg_limones = ?
                WHERE rowid = ?
            """, (colaboradores_nuevos, kg_tortilla_nuevos, kg_queso_nuevos, kg_tortilla_harina_nuevos,
                  kg_bistek_nuevos, kg_chorizo_nuevos, kg_pastor_nuevos, kg_cebolla_nuevos, kg_limones_nuevos,
                  rowid_taquiza))
            conexion.commit()
            messagebox.showinfo("Éxito", "Los cambios se han guardado correctamente.")
            ventana_detalles.destroy()
        else:
            messagebox.showerror("Error", "No se pudo conectar a la base de datos.")
    except ValueError:
        messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos para los insumos.")



# Función para ordenar el Treeview
def ordenar_treeview(columna, reverso):
    datos = [(tree.set(item, columna), item) for item in tree.get_children('')]
    datos.sort(reverse=reverso)
    
    for index, (valor, item) in enumerate(datos):
        tree.move(item, '', index)

    tree.heading(columna, command=lambda: ordenar_treeview(columna, not reverso))


# Crear botones
boton_mostrar_registros = tk.Button(ventana_principal, text="Mostrar Registros", command=mostrar_registros)
boton_mostrar_registros.pack(side=tk.LEFT)

boton_mostrar_estadisticas = tk.Button(ventana_principal, text="Mostrar Estadísticas", command=mostrar_estadisticas)
boton_mostrar_estadisticas.pack(side=tk.LEFT)

boton_mostrar_graficas = tk.Button(ventana_principal, text="Mostrar Gráficas", command=mostrar_graficas_estadisticas)
boton_mostrar_graficas.pack(side=tk.LEFT)

boton_analizar_series = tk.Button(ventana_principal, text="Analizar Series Temporales", command=analizar_series_temporales)
boton_analizar_series.pack(side=tk.LEFT)

boton_clustering = tk.Button(ventana_principal, text="Realizar Clustering", command=realizar_clustering)
boton_clustering.pack(side=tk.LEFT)

# Crear un botón para realizar el análisis de sentimientos
boton_analizar = tk.Button(ventana_principal, text="Analizar Sentimientos", command=analizar_sentimientos)
boton_analizar.pack(side=tk.LEFT)

# Llamar a la función para agregar el botón en la ventana principal
boton_asignacion = tk.Button(ventana_principal, text="Asignación para Evento", command=asignar_evento)
boton_asignacion.pack(side=tk.LEFT)

boton_modificar = tk.Button(ventana_principal, text="Modificar Registro", command=modificar_registro)
boton_modificar.pack(pady=10)

boton_eliminar = tk.Button(ventana_principal, text="Eliminar Registro", command=eliminar_registro)
boton_eliminar.pack(pady=10)

# Crear botón para abrir la ventana de agregar registro
boton_agregar = tk.Button(ventana_principal, text="Agregar Registro", command=agregar_nuevo_registro)
boton_agregar.pack(pady=10)

ventana_principal.geometry("800x600")
ventana_principal.mainloop()
