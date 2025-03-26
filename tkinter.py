import json
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import pipeline
import re
import tkinter as tk
from tkinter import messagebox, scrolledtext

# Cargar datos de celulares
def load_data():
    with open("celulares_con_benchmarks.json", "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

data = load_data()
data = data.drop(columns=["device", "company"], errors='ignore')
data = data.dropna()

# Cargar criterios de evaluación
def load_criteria():
    with open("criterios.json", "r", encoding="utf-8") as f:
        return json.load(f)["criterios"]

criterios = load_criteria()

# Obtener lista de marcas disponibles
marcas_disponibles = data["Company Name"].unique()

def nombres_celulares(nombre):
    nombres_guardados = nombre
    return nombres_guardados 

# Normalizar datos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))

# Crear red neuronal para recomendación
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(data_scaled.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
y_train = np.random.rand(len(data_scaled))  # Simulación de satisfacción del usuario
model.fit(data_scaled, y_train, epochs=50, verbose=0)

# Modelo preentrenado para NLP
nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Cargar historial de consultas previas
def load_historial():
    try:
        with open("historial_consultas.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

historial = load_historial()

# Función principal para recomendar celulares
def recomendar_celular():
    user_input = entry_input.get().strip()
    
    if user_input.lower() == "salir":
        window.quit()
        return
    
    # Revisar historial antes de usar el modelo NLP
    for consulta in historial:
        if consulta["consulta"].lower() == user_input.lower():
            text_output.delete(1.0, tk.END)
            text_output.insert(tk.END, "📌 Encontré una recomendación previa para esta consulta:\n")
            for resultado in consulta["resultados"]:
                text_output.insert(tk.END, f"📱 {resultado['Company Name']} {resultado['Model Name']} - {resultado['Launched Price (USA)']}$\n")
                text_output.insert(tk.END, f"   🔋 Batería: {resultado['Battery Capacity (mAh)']}mAh | 📸 Cámara: {resultado['Back Camera (MP)']}MP\n")
                text_output.insert(tk.END, f"   🎮 GPU: {resultado['gpuScore']} | 🚀 CPU: {resultado['cpuScore']} | 📏 Pantalla: {resultado['Screen Size (inches)']}\n")
            return
    
    # Detectar intenciones del usuario
    result = nlp_model(user_input, candidate_labels=list(criterios.keys()))
    best_matches = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.1]
    
    if not best_matches:
        messagebox.showinfo("Resultado", "❌ No entendí bien qué buscas. ¿Podrías darme más detalles?")
        return
    
    text_output.delete(1.0, tk.END)
    text_output.insert(tk.END, f"📌 Buscando celulares con énfasis en: {', '.join(best_matches)}...\n")
    
    opciones_mostradas = set()
    consultas_realizadas = []
    
    # Obtener los criterios correspondientes
    columnas = []
    asc = False
    
    for match in best_matches:
        criterio = criterios[match]
        if isinstance(criterio["columna"], list):
            columnas.extend(criterio["columna"])
        else:
            columnas.append(criterio["columna"])
        if "ascendente" in criterio:
            asc = criterio["ascendente"]
    
    data_filtrada = data
    filtered = data_filtrada.sort_values(by=columnas, ascending=asc)
    filtered = filtered[~filtered.index.isin(opciones_mostradas)].head(3)
    
    if filtered.empty:
        messagebox.showinfo("Resultado", "❌ Lo siento, no encontré más opciones con esas características.")
        return
    
    text_output.insert(tk.END, "✨ Aquí tienes tres opciones que podrían interesarte:\n")
    for idx, row in filtered.iterrows():
        opciones_mostradas.add(idx)
        text_output.insert(tk.END, f"📱 {row['Company Name']} {row['Model Name']} - {row['Launched Price (USA)']}$\n")
        text_output.insert(tk.END, f"   🔋 Batería: {row['Battery Capacity (mAh)']}mAh | 📸 Cámara: {row['Back Camera (MP)']}MP\n")
        text_output.insert(tk.END, f"   🎮 GPU: {row['gpuScore']} | 🚀 CPU: {row['cpuScore']} | 📏 Pantalla: {row['Screen Size (inches)']}\n")
    
    consultas_realizadas.append({"consulta": user_input, "criterios": best_matches, "resultados": filtered.to_dict(orient="records")})
    
    historial.append(consultas_realizadas[-1])
    with open("historial_consultas.json", "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=4, ensure_ascii=False)
    
    messagebox.showinfo("Resultado", "🎉 ¡Aquí están tus recomendaciones!")

# Configuración de la ventana principal con Tkinter
window = tk.Tk()
window.title("Recomendador de Celulares")
window.geometry("600x400")

# Etiqueta y caja de texto de entrada
label_input = tk.Label(window, text="¿Qué tipo de celular estás buscando?")
label_input.pack(pady=10)

entry_input = tk.Entry(window, width=50)
entry_input.pack(pady=10)

# Botón para obtener recomendaciones
btn_recomendar = tk.Button(window, text="Buscar", command=recomendar_celular)
btn_recomendar.pack(pady=10)

# Área de salida con scroll
text_output = scrolledtext.ScrolledText(window, width=70, height=15)
text_output.pack(pady=10)

# Iniciar la interfaz
window.mainloop()
