import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import io
import os
from pymongo import MongoClient
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# CONECTAMOS CON MONGO DB 
MONGO_URI = os.getenv("MONGO_URI") 
coleccion = None

try:
    if MONGO_URI is None:
        raise ValueError("MONGO_URI no está definido en HuggingFace Secrets. ¡No se guardarán datos!")
    
    cliente = MongoClient(MONGO_URI)
    db = cliente["muchacham"]
    coleccion = db["candidatos"] 
    print("Conexión a MongoDB exitosa.")

except Exception as e:
    print(f"Error MongoDB al iniciar: {e}")


#UTILIZAMOS CIFRADO RSA 
def generar_claves_rsa():
    """Genera y retorna el par de claves RSA."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048 
    )
    public_key = private_key.public_key()
    return private_key, public_key 

PRIVATE_KEY, PUBLIC_KEY = generar_claves_rsa()

def cifrar_rsa(data, public_key):
    """Cifra un string de datos con la clave pública."""
    if not isinstance(data, str):
        data = str(data)
    
    try:
        cifrado = public_key.encrypt(
            data.encode('utf-8'),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return cifrado.hex() 
    except ValueError as e:
        return f"Error: Dato demasiado largo para cifrado RSA. {e}"


# PREGUNTAS/PUESTOS
PUESTOS = [
    "CARGADOR",
    "VENTAS/COBRANZA",
    "REPARTIDOR",
    "LIMPIEZA",
    "ADMINISTRATIVO"
]

NUM_PREGUNTAS = 15

# Diccionario de preguntas específicas para cada puesto
PREGUNTAS = {
    "CARGADOR": [
        "¿Trabajas bien bajo presión?", "¿Mantienes la calma con carga pendiente?",
        "¿Te comunicas antes de mover mercancía pesada?", "¿Revisas la mercancía antes de moverla?",
        "¿Sigues indicaciones de seguridad al levantar objetos?", "¿Organizas tu área para evitar accidentes?",
        "¿Te adaptas fácilmente a cambios?", "¿Trabajas bien en equipo?",
        "¿Detectas mercancía dañada?", "¿Has usado patín hidráulico?",
        "¿Respetas descansos permitidos?", "¿Eres responsable con tus tiempos?",
        "¿Mantienes actitud positiva?", "¿Reportas irregularidades de inmediato?",
        "¿Sigues instrucciones sin supervisión?"
    ],
    "VENTAS/COBRANZA": [
        "¿Te sientes cómodo hablando con clientes?", "¿Mantienes la calma con clientes molestos?",
        "¿Persistente sin exagerar?", "¿Llevas control de cuentas?",
        "¿Te adaptas a trabajar con metas?", "¿Negocias sin conflicto?",
        "¿Analizas riesgos de pago?", "¿Te comunicas bien por teléfono?",
        "¿Eres paciente?", "¿Has usado sistemas o facturación?",
        "¿Detectas necesidades del cliente?", "¿Aceptas un no sin frustrarte?",
        "¿Avisas atrasos?", "¿Te organizas con varias cuentas?",
        "¿Recuperas pagos atrasados?"
    ],
    "REPARTIDOR": [
        "¿Usas GPS adecuadamente?", "¿Eres puntual?",
        "¿Controlas el estrés con muchas entregas?", "¿Verificas mercancía antes de ruta?",
        "¿Actitud amable con clientes?", "¿Reportas contratiempos?",
        "¿Sigues rutas nuevas sin problema?", "¿Conduces distancias largas sin perder concentración?",
        "¿Experiencia como repartidor?", "¿Cuidas la mercancía?",
        "¿Te adaptas a cambios?", "¿Atención a clientes molestos?",
        "¿Administras dinero/comprobantes?", "¿Conoces normas de tránsito?",
        "¿Aseguras el vehículo?"
    ],
    "LIMPIEZA": [
        "¿Eres organizada?", "¿Trabajas sin supervisión?",
        "¿Mantienes calma con urgencias?", "¿Sigues protocolos de limpieza?",
        "¿Usas productos correctamente?", "¿Reportas objetos fuera de lugar?",
        "¿Trabajas sin interrumpir a otros?", "¿Tareas repetitivas OK?",
        "¿Cuidas químicos o materiales?", "¿Priorizas tareas?",
        "¿Seguridad en áreas riesgosas?", "¿Aceptas retroalimentación?",
        "¿Levantas objetos ligeros?", "¿Discreción con objetos sensibles?",
        "¿Puntual?"
    ],
    "ADMINISTRATIVO": [
        "¿Eres organizado?", "¿Manejas varias tareas a la vez?",
        "¿Sabes usar software de oficina?", "¿Toleras presión?",
        "¿Revisas tu trabajo antes de entregarlo?", "¿Pides info cuando falta?",
        "¿Calma con jefes molestos?", "¿Te adaptas a cambios?",
        "¿Manejas información confidencial?", "¿Sigues instrucciones al pie?",
        "¿Captura repetitiva OK?", "¿Detectas errores antes de enviar?",
        "¿Puntual?", "¿Buena convivencia con áreas?",
        "¿Trabajas sin supervisión?"
    ]
}

# ENTRENAMIENTO SIMPLE DEL MODELO
def generar_datos():
    X, y = [], []
    for idx, puesto_nombre in enumerate(PUESTOS):
        for _ in range(300):
            X.append(np.random.randint(0, 2, NUM_PREGUNTAS))
            y.append(idx)
    return np.array(X), np.array(y)

X, y = generar_datos()
scaler = StandardScaler().fit(X)
modelo = MLPClassifier(hidden_layer_sizes=(20,10), max_iter=400, random_state=42).fit(scaler.transform(X), y)

# PARA GUARDAR LA INFO EN MONGO CON RSA
def guardar_en_mongo(datos):
    global coleccion
    if coleccion is not None:
        try:
            datos_sensibles = ["nombre", "edad", "telefono", "correo"]
            datos_para_guardar = datos.copy()
            for campo in datos_sensibles:
                if campo in datos_para_guardar:
                    datos_para_guardar[campo] = cifrar_rsa(datos_para_guardar[campo], PUBLIC_KEY)
                    
            coleccion.insert_one(datos_para_guardar)
            return "Datos (sensibles cifrados con RSA) guardados exitosamente en MongoDB."
        except Exception as e:
            return f"Error al guardar en MongoDB: {str(e)}"
    else:
        return "ADVERTENCIA: No se pudo guardar. Conexión a MongoDB fallida."

# PARA EL PDF
def generar_pdf(nombre, puesto, recomendado):
    ruta = "/tmp/resultado.pdf"
    doc = SimpleDocTemplate(ruta, pagesize=letter)
    styles = getSampleStyleSheet()
    
    contenido = [
        Paragraph("<b><font size='14'>REPORTE DE EVALUACIÓN LABORAL</font></b>", styles["Title"]),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph(f"<b>Nombre del Candidato:</b> {nombre}", styles["BodyText"]),
        Paragraph(f"<b>Puesto Seleccionado:</b> {puesto}", styles["BodyText"]),
        Paragraph(f"<b>Recomendación del Modelo (IA):</b> <font color='red'>{recomendado}</font>", styles["BodyText"]),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph("<i>Este reporte se basa en las respuestas proporcionadas al modelo de clasificación.</i>", styles["BodyText"])
    ]
    
    doc.build(contenido)
    return ruta

# PARA EVALUAR AL CNDIDATO
def evaluar(nombre, edad, telefono, correo, puesto, *respuestas):
    try:
        try:
            edad = str(int(edad)) 
        except ValueError:
            return "Error: La edad debe ser un número.", None, None

        if int(edad) < 18:
            return "Debes ser mayor de edad para aplicar.", None, None

        if len(respuestas) != NUM_PREGUNTAS or any(r not in ["S","N"] for r in respuestas):
            return "Error: Debes responder las 15 preguntas (S/N).", None, None
        vector_respuestas = np.array([1 if r=="S" else 0 for r in respuestas]).reshape(1,-1)
        probas = modelo.predict_proba(scaler.transform(vector_respuestas))[0]
        recomendado = PUESTOS[np.argmax(probas)]
        datos_candidato = {
            "fecha_evaluacion": datetime.now().isoformat(),
            "nombre": nombre,
            "edad": edad, 
            "telefono": telefono,
            "correo": correo,
            "puesto_aplicado": puesto,
            "respuestas_binarias": [1 if r=="S" else 0 for r in respuestas],
            "puesto_recomendado": recomendado,
            "probabilidades": {PUESTOS[i]: float(probas[i]) for i in range(len(PUESTOS))}
        }
        mensaje_mongo = guardar_en_mongo(datos_candidato)

        # GENERAMOS UNA GRAFICA SENCILLA
        fig, ax = plt.subplots(figsize=(8, 5))
        colores = ['#4CAF50' if p == recomendado else '#FF5733' for p in PUESTOS]
        
        ax.bar(PUESTOS, probas, color=colores)
        ax.set_title("Probabilidad de Adecuación por Puesto")
        ax.set_ylabel("Probabilidad (0.0 - 1.0)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        #IMPRIMIMOS EL PDF
        pdf_path = generar_pdf(nombre, puesto, recomendado)
        resultado_final = f"Recomendado: {recomendado}\n[{mensaje_mongo}]"
        return resultado_final, fig, pdf_path

    except Exception as e:
        return f"Error interno en Evaluación: {str(e)}", None, None

# DEFINIMOS LA INTERFAZ EN GRADIO
def interfaz():
    with gr.Blocks(title="Evaluación Laboral") as demo:
        gr.Markdown("## Evaluación de Candidatos — Recomendación por IA (con MongoDB)")
        gr.Markdown("Completa los datos. **Los datos sensibles (nombre, edad, teléfono, correo) serán cifrados con RSA** antes de guardarse en MongoDB.")
        with gr.Row():
            nombre = gr.Textbox(label="1. Nombre del Candidato", scale=2)
            edad = gr.Number(label="2. Edad", minimum=18, scale=1)
        with gr.Row():
            telefono = gr.Textbox(label="3. Teléfono", scale=1)
            correo = gr.Textbox(label="4. Correo Electrónico", scale=2)
        
        puesto = gr.Dropdown(PUESTOS, label="5. Puesto a Evaluar", info="Selecciona el puesto para cargar las 15 preguntas específicas.", value=PUESTOS[0])
        gr.Markdown("### 6. Preguntas de Evaluación (Responde SÍ (S) o NO (N))")
        
        radios = [
            gr.Radio(
                ["S", "N"], 
                label=PREGUNTAS[PUESTOS[0]][i], 
                value="", 
                interactive=True,
                scale=1
            ) 
            for i in range(NUM_PREGUNTAS)
        ]

        def update_questions(puesto_seleccionado):
            if puesto_seleccionado in PREGUNTAS:
                return [gr.update(label=PREGUNTAS[puesto_seleccionado][i], value="") for i in range(NUM_PREGUNTAS)]
            return [gr.update(label=f"Selecciona un puesto válido.", value="") for i in range(NUM_PREGUNTAS)]

        puesto.change(
            update_questions, 
            inputs=puesto, 
            outputs=radios,
            queue=False
        )

        btn = gr.Button("Evaluar y Guardar Resultados (Cifrados)", variant="primary")
        
        gr.Markdown("### Resultados de la Recomendación y Estado de Guardado")
        with gr.Row():
            out1 = gr.Textbox(label="Resultado (Recomendación y MongoDB)", lines=3, scale=1)
            out3 = gr.File(label="Descargar PDF del Reporte", scale=1)
        
        out2 = gr.Plot(label="Gráfica de Probabilidades")

        btn.click(
            evaluar, 
            inputs=[nombre, edad, telefono, correo, puesto] + radios, 
            outputs=[out1, out2, out3],
            queue=True
        )
    return demo

if __name__ == "__main__":
    interfaz().launch()