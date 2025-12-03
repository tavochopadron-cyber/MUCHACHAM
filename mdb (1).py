import base64
import secrets
from collections import defaultdict
from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
MONGO_URI = "mongodb+srv://tavo_user:1234567%23@cluster0.cjx2xfb.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["muchacham_db"]
coleccion = db["candidatos"]
NUM_PREGUNTAS_POR_PUESTO = 15
PUESTOS = ["CARGADOR", "VENTAS/COBRANZA", "REPARTIDOR", "LIMPIEZA", "ADMINISTRATIVO"]
PREGUNTAS = {
    "CARGADOR": [
        "¿Te consideras una persona que trabaja bien bajo presión?",
        "¿Mantienes la calma cuando hay mucha carga pendiente?",
        "¿Te comunicas con tu equipo antes de mover mercancía pesada?",
        "¿Sueles revisar la mercancía antes de moverla?",
        "¿Sigues las indicaciones de seguridad al levantar objetos?",
        "¿Sueles organizar tu área de trabajo para evitar accidentes?",
        "¿Te adaptas fácilmente cuando cambian las prioridades del día?",
        "¿Puedes trabajar en equipo sin conflictos?",
        "¿Detectas a tiempo mercancía dañada o mal etiquetada?",
        "¿Tienes experiencia usando patín hidráulico?",
        "¿Tomas descansos solo cuando están permitidos?",
        "¿Te consideras responsable con los tiempos asignados?",
        "¿Puedes mantener una actitud positiva en días complicados?",
        "¿Avisas de inmediato cuando encuentras alguna irregularidad?",
        "¿Sabes seguir instrucciones sin supervisión constante?"
    ],
    "VENTAS/COBRANZA": [
        "¿Te sientes cómodo hablando con clientes diariamente?",
        "¿Puedes mantener la calma cuando un cliente está molesto?",
        "¿Sueles ser persistente sin caer en insistencia excesiva?",
        "¿Llevas un control ordenado de tus cuentas o visitas?",
        "¿Te adaptas bien a trabajar con metas?",
        "¿Eres capaz de negociar sin generar conflicto?",
        "¿Analizas si un cliente representa riesgo de pago?",
        "¿Puedes comunicarte con claridad por teléfono?",
        "¿Te consideras una persona paciente?",
        "¿Has usado sistemas de registro o facturación?",
        "¿Sueles detectar las necesidades del cliente rápidamente?",
        "¿Puedes aceptar un “no” sin frustrarte?",
        "¿Informas a tiempo cuando detectas un atraso en un cliente?",
        "¿Te mantienes organizado aun con muchas cuentas simultáneas?",
        "¿Te sientes capaz de recuperar pagos atrasados?"
    ],
    "REPARTIDOR": [
        "¿Te orientas bien usando GPS o mapas digitales?",
        "¿Te consideras una persona puntual?",
        "¿Puedes manejar bien el estrés cuando las entregas aumentan?",
        "¿Verificas la mercancía antes de salir a ruta?",
        "¿Mantienes una actitud amable en todas tus entregas?",
        "¿Reportas de inmediato cualquier contratiempo en ruta?",
        "¿Sigues rutas nuevas sin problema?",
        "¿Puedes manejar distancias largas sin perder concentración?",
        "¿Has manejado antes un vehículo de reparto?",
        "¿Cuidas la mercancía para que llegue en buen estado?",
        "¿Te adaptas rápido cuando hay cambios de último momento?",
        "¿Puedes tratar con clientes molestos sin perder la calma?",
        "¿Administra bien el dinero o comprobantes durante las entregas?",
        "¿Conoces normas básicas de tránsito?",
        "¿Tomas medidas para asegurar el vehículo durante la ruta?"
    ],
    "LIMPIEZA": [
        "¿Te consideras una persona organizada?",
        "¿Puedes trabajar sin supervisión directa?",
        "¿Sueles mantener una actitud tranquila cuando te piden limpiar algo urgente?",
        "¿Puedes seguir protocolos de limpieza establecidos?",
        "¿Sabes usar productos básicos de limpieza?",
        "¿Informas cuando encuentras algo fuera de lugar?",
        "¿Puedes trabajar en áreas donde hay otras personas sin interrumpirlas?",
        "¿Te adaptas bien a tareas repetitivas?",
        "¿Cuidas el uso de químicos o materiales?",
        "¿Priorizas las tareas cuando hay muchas pendientes?",
        "¿Sigues medidas de seguridad al limpiar áreas riesgosas?",
        "¿Aceptas retroalimentación sin problema?",
        "¿Puedes levantar objetos ligeros o mover mobiliario pequeño?",
        "¿Mantienes discreción cuando encuentras información u objetos sensibles?",
        "¿Te consideras una persona puntual y constante?"
    ],
    "ADMINISTRATIVO": [
        "¿Te consideras una persona organizada?",
        "¿Puedes trabajar con varias tareas al mismo tiempo?",
        "¿Tienes experiencia usando computadoras o software de oficina?",
        "¿Te molesta trabajar bajo presión?",
        "¿Verificas tu trabajo antes de entregarlo para evitar errores?",
        "¿Comunicas a tiempo cuando te falta información para avanzar?",
        "¿Puedes mantener la calma con jefes o usuarios molestos?",
        "¿Te adaptas fácilmente cuando cambian prioridades?",
        "¿Puedes manejar información confidencial con responsabilidad?",
        "¿Sigues instrucciones con precisión?",
        "¿Te sientes cómodo haciendo capturas o registros repetitivos?",
        "¿Sueles detectar errores en documentos antes de enviarlos?",
        "¿Te consideras una persona puntual?",
        "¿Te llevas bien con diferentes áreas de trabajo?",
        "¿Puedes trabajar sin supervisión directa?"
    ]
}
def generar_datos_sinteticos(n_por_clase=400, seed=42):
    np.random.seed(seed)
    X = []
    y = []
    for idx, puesto in enumerate(PUESTOS):
        for _ in range(n_por_clase):
            vect = np.zeros(NUM_PREGUNTAS_POR_PUESTO * len(PUESTOS), dtype=int)
            for j, p in enumerate(PUESTOS):
                start = j * NUM_PREGUNTAS_POR_PUESTO
                if p == puesto:
                    probs = np.full(NUM_PREGUNTAS_POR_PUESTO, 0.78)
                else:
                    probs = np.full(NUM_PREGUNTAS_POR_PUESTO, 0.22)
                sampled = (np.random.rand(NUM_PREGUNTAS_POR_PUESTO) < probs).astype(int)
                vect[start:start + NUM_PREGUNTAS_POR_PUESTO] = sampled
            X.append(vect)
            y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y
def entrenar_o_cargar_modelo(path_modelo="modelo_muchacham.pkl"):
    try:
        modelo, scaler = joblib.load(path_modelo)
        return modelo, scaler
    except:
        X, y = generar_datos_sinteticos()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        modelo = MLPClassifier(hidden_layer_sizes=(60,30), activation="relu", max_iter=500, random_state=1)
        modelo.fit(X_train_s, y_train)
        joblib.dump((modelo, scaler), path_modelo)
        return modelo, scaler
modelo, scaler = entrenar_o_cargar_modelo()
def respuestas_a_vector(respuestas_por_puesto):
    vect = []
    for puesto in PUESTOS:
        atre = respuestas_por_puesto.get(puesto, [])
        if len(atre) != NUM_PREGUNTAS_POR_PUESTO:
            atre = ["N"] * NUM_PREGUNTAS_POR_PUESTO
        vect.extend([1 if r.upper().strip() in ("S","SI","YES","Y") else 0 for r in atre])
    return np.array(vect, dtype=int)
class RSAKeySimulada:
    def __init__(self):
        self.public_key = base64.b64encode(secrets.token_bytes(32)).decode("utf-8")
        self.private_key = base64.b64encode(secrets.token_bytes(32)).decode("utf-8")
def cifrar_rsa_simulado(datos, clave_publica):
    datos_codificados = base64.urlsafe_b64encode(datos.encode("utf-8")).decode("utf-8")
    return f"RSA-256:{clave_publica[:16]}...{datos_codificados[:40]}..."
def pedir_datos_personales():
    def pedir(m):
        while True:
            x = input(m).strip()
            if x:
                return x
            print("No puede quedar vacío.")
    print("\n--- DATOS PERSONALES ---")
    return {
        "nombre": pedir("Nombre completo: "),
        "edad": pedir("Edad: "),
        "sexo": pedir("Sexo (M/F/Otro): "),
        "correo": pedir("Correo electrónico (se cifrará): "),
        "telefono": pedir("Número de teléfono (se cifrará): ")
    }
def realizar_cuestionario_grupal():
    respuestas = {}
    for puesto in PUESTOS:
        print(f"\n--- PREGUNTAS PARA {puesto} ---")
        rlist = []
        for i, p in enumerate(PREGUNTAS[puesto]):
            while True:
                r = input(f"{i+1}. {p} (S/N): ").upper().strip()
                if r in ("S","N","SI","NO","Y","YES"):
                    rlist.append("Sí" if r.startswith("S") or r in ("Y","YES") else "No")
                    break
                print("Respuesta inválida.")
        respuestas[puesto] = rlist
    return respuestas
def predecir_puesto(respuestas_por_puesto):
    v = respuestas_a_vector(respuestas_por_puesto).reshape(1, -1)
    v_s = scaler.transform(v)
    probs = modelo.predict_proba(v_s)[0]
    idx = int(np.argmax(probs))
    return PUESTOS[idx], float(probs[idx]), {PUESTOS[i]: float(p) for i,p in enumerate(probs)}
def iniciar_proceso_seleccion():
    print("\n========================================")
    print("    🌟 SISTEMA DE SELECCIÓN MUCHACHAM 🌟")
    print("========================================")
    datos = pedir_datos_personales()
    respuestas_por_puesto = realizar_cuestionario_grupal()
    key = RSAKeySimulada()
    datos_cifrados = cifrar_rsa_simulado(f"Email:{datos['correo']}|Tel:{datos['telefono']}", key.public_key)
    puesto_predicho, prob, probs_detalle = predecir_puesto(respuestas_por_puesto)
    aceptado = prob >= 0.5
    registro = {
        "nombre": datos["nombre"],
        "edad": datos["edad"],
        "sexo": datos["sexo"],
        "correo_cifrado": datos_cifrados,
        "respuestas_por_puesto": respuestas_por_puesto,
        "puesto_predicho": puesto_predicho,
        "probabilidad_puesto_predicho": prob,
        "probabilidades": probs_detalle,
        "aceptado_modelo": aceptado
    }
    coleccion.insert_one(registro)
    print("\n Datos guardados en MongoDB.\n")
    print("=========== RESULTADO FINAL ===========")
    print(f"Puesto recomendado por la red: {puesto_predicho} ({prob*100:.2f}%)")
    print("Probabilidades por puesto:")
    for p, pv in probs_detalle.items():
        print(f" - {p}: {pv*100:.2f}%")
    print("Estado:", "ACEPTADO ✅" if aceptado else "NO ACEPTADO ❌")
    print("========================================")
if __name__ == "__main__":
    iniciar_proceso_seleccion()
