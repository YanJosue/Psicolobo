import spacy
import pandas as pd
from fuzzywuzzy import fuzz
from collections import defaultdict


# --- 0. Cargar la Base de Conocimiento 

def cargar_diccionario_emociones(ruta_csv='baseConocimientos/emociones_v2.csv'):
    """
    Carga el diccionario de emociones desde un archivo CSV.
    Retorna un DataFrame de pandas.
    """
   

    try:
        # MUY IMPORTANTE:'sep=',' ya que CSV usa comas. 
        df = pd.read_csv(ruta_csv, encoding='utf-8', sep=',') 

        # LÍNEA DE DEPURACIÓN (para ver qué columnas lee Pandas) 
        print(f"DEBUG: Nombres de columnas detectadas por pandas: {df.columns.tolist()}")
        

        print(f"Diccionario de emociones cargado exitosamente desde '{ruta_csv}'.")
        return df
    except Exception as e:
        print(f"Error al cargar el diccionario CSV: {e}")
        print("Asegúrate de que el CSV esté bien formado y con la codificación correcta (UTF-8).")
        print("También, confirma que el 'sep=' en pd.read_csv coincide con el delimitador real de tu CSV.")
        return None

# Cargar el diccionario al inicio 
diccionario_emociones_df = cargar_diccionario_emociones()


# Si el diccionario no se carga, el script no puede continuar
if diccionario_emociones_df is None:
    print("No se pudo cargar el diccionario de emociones. Terminando la ejecución.")
    exit()

# Cargar el modelo de spaCy una vez
try:
    nlp = spacy.load("es_core_news_sm")
    print("Modelo de spaCy 'es_core_news_sm' cargado exitosamente.")
except OSError:
    print("El modelo 'es_core_news_sm' de spaCy no está descargado.")
    print("Por favor, ejecuta: python -m spacy download es_core_news_sm")
    exit()

# Definir palabras de negación 
PALABRAS_NEGACION = [
    "no", "nunca", "jamás", "ni", "tampoco", "nada", "nadie", "ninguno",
    "para nada", "ni madres", "ni de chiste", "para nada" # Regionalismos/coloquialismos
]

# --- Función Principal del Algoritmo de Detección de Emociones ---
def detectar_emocion(texto_usuario: str, diccionario_df: pd.DataFrame, umbral_difuso: int = 80):
    """
    Implementa el algoritmo de detección de emociones basado en léxico de SERMO.
    Retorna la emoción detectada y el conteo de términos por emoción.
    """
    if diccionario_df is None:
        print("Error interno: El diccionario de emociones no fue cargado en detectar_emocion.")
        return "ERROR_DIC_NO_CARGADO", {}, {}

    # 1. Sentence segmentation
    doc = nlp(texto_usuario)
    oraciones = [sent.text for sent in doc.sents]
    print(f"\nOraciones detectadas: {oraciones}")

    # Inicializar conteo de emociones
    conteo_emociones = defaultdict(int)
    terminos_encontrados_por_emocion = defaultdict(list)

    for oracion in oraciones:
        doc_oracion = nlp(oracion)
        # Convertimos a minúsculas y obtenemos lemas para una mejor coincidencia.
        # Filtramos stop words y no-alfabéticos al inicio, pero con cuidado.
        # NOTA: Los lemmas son una forma más robusta de emparejar palabras.
        # Por ejemplo, "enojado", "enojar", "enojan" -> "enojar"
        # Esto reduce la necesidad de tantos sinónimos si el lema es suficiente.
        
        # Guardaremos los tokens originales con sus lemas para el procesamiento
        processed_tokens = []
        for token in doc_oracion:
            # Incluye solo tokens alfabéticos (que también incluye caracteres con tilde, eñes)
            # Excluye stop words para reducir ruido, pero cuidado con frases idiomáticas.
            if token.is_alpha and not token.is_stop:
                processed_tokens.append({'original': token.text, 'lemma': token.lemma_.lower(), 'is_negated': False, 'spacy_token_obj': token})

        print(f"Tokens procesados (lemas) en la oración '{oracion}': {[t['lemma'] for t in processed_tokens]}")

        # --- Lógica de Negación (revisada) ---
        # Marcar tokens individuales como negados si están cerca de una palabra de negación.
        for i, p_token in enumerate(processed_tokens):
            token_obj = p_token['spacy_token_obj']
            # Obtener el contexto de 3 palabras antes del token actual en la oración original
            start_idx = max(0, token_obj.i - 3)
            contexto_previo_original = [t.text.lower() for t in doc_oracion[start_idx:token_obj.i]]
            
            if any(neg_word in contexto_previo_original for neg_word in PALABRAS_NEGACION):
                p_token['is_negated'] = True
                print(f"  -> '{p_token['original']}' (lema: '{p_token['lemma']}') marcado como negado debido a contexto: {contexto_previo_original}")


        # 5. Emotion recognition (Lexicon-based con Fuzzy Matching)
        for p_token in processed_tokens: # Iterar sobre los tokens procesados
            token_original_text = p_token['original']
            token_lemma = p_token['lemma']
            
            # Si este token fue marcado como negado Y el término en el diccionario debe considerar negación, sáltalo.
            # Verificamos si la bandera 'is_negated' es True para este token específico
            # y si el término del diccionario también tiene Negacion_Considerar = 'sí'.
            # La verificación de 'Negacion_Considerar' se hará dentro del bucle del diccionario
            # porque depende de la fila del diccionario.

            # Iterar sobre el diccionario para encontrar coincidencias
            for idx, row in diccionario_df.iterrows():
                termino_emocional_dic = str(row['Termino_emocional']).lower()
                lista_sinonimos_dic = str(row['Lista_sinonimos']).lower().split(', ') # Asumiendo "sinonimo1, sinonimo2"
                emocion = str(row['Emocion']).lower()
                negacion_considerar_str = str(row['Negacion_Considerar']).lower()

                # --- Verificación de negación antes de la coincidencia ---
                # Si el token actual está negado Y el término del diccionario *debe* considerar negación,
                # entonces no lo contamos y pasamos al siguiente.
                if p_token['is_negated'] and negacion_considerar_str == 'sí':
                   
                    # print(f"  -> Término '{token_original_text}' (lema: '{token_lemma}') ignorado debido a negación para '{termino_emocional_dic}'.")
                    continue # Salta esta combinación token-diccionario

                # Prioridad 1: Coincidencia exacta con el término emocional del diccionario
                if token_lemma == termino_emocional_dic:
                    conteo_emociones[emocion] += 1
                    terminos_encontrados_por_emocion[emocion].append(token_original_text)
                    print(f"  -> Coincidencia EXACTA (lema): '{token_lemma}' ('{token_original_text}') -> {emocion} (vs '{termino_emocional_dic}')")
                    break # Salir del bucle interno (del diccionario), ya encontramos una coincidencia para este token

                # Prioridad 2: Coincidencia exacta con algún sinónimo en la lista
                if token_lemma in lista_sinonimos_dic:
                    conteo_emociones[emocion] += 1
                    terminos_encontrados_por_emocion[emocion].append(token_original_text)
                    print(f"  -> Coincidencia EXACTA (sinónimo): '{token_lemma}' ('{token_original_text}') -> {emocion} (en sinónimos de '{termino_emocional_dic}')")
                    break 

                # Prioridad 3: Coincidencia difusa con el término emocional (solo si no hubo coincidencia exacta)
                # NOTA: La coincidencia difusa debe ser la última opción
                similitud_termino = fuzz.ratio(token_lemma, termino_emocional_dic)
                if similitud_termino >= umbral_difuso:
                    conteo_emociones[emocion] += 1
                    terminos_encontrados_por_emocion[emocion].append(token_original_text)
                    print(f"  -> Coincidencia DIFUSA ({similitud_termino}%): '{token_lemma}' ('{token_original_text}') con '{termino_emocional_dic}' -> {emocion}")
                    break # Salir del bucle interno

                # Prioridad 4: Coincidencia difusa con algún sinónimo
                # Esto es más intensivo computacionalmente, considéralo si el rendimiento es un problema.
                # Puedes quitarlo si la detección es lo suficientemente buena con solo los términos principales.
                for sinonimo_dic in lista_sinonimos_dic:
                    similitud_sinonimo = fuzz.ratio(token_lemma, sinonimo_dic)
                    if similitud_sinonimo >= umbral_difuso:
                        conteo_emociones[emocion] += 1
                        terminos_encontrados_por_emocion[emocion].append(token_original_text)
                        print(f"  -> Coincidencia DIFUSA (sinónimo, {similitud_sinonimo}%): '{token_lemma}' ('{token_original_text}') con sinónimo '{sinonimo_dic}' de '{termino_emocional_dic}' -> {emocion}")
                        break # Si encuentra un sinónimo difuso, pasa al siguiente token del usuario
                else: # Si el bucle interno de sinónimos terminó sin un 'break'
                    continue # Continúa con el siguiente término del diccionario si no se encontró nada


    # 6. Voting
    if not conteo_emociones or all(v == 0 for v in conteo_emociones.values()):
        emocion_detectada = "NO_DETECTADA"
    else:
        emocion_detectada = max(conteo_emociones, key=conteo_emociones.get)
        max_conteo = conteo_emociones[emocion_detectada]
        # Verificar si hay empate
        if list(conteo_emociones.values()).count(max_conteo) > 1:
            emocion_detectada = "EMPATE" # O podrías devolver una lista de emociones empatadas

    return emocion_detectada, conteo_emociones, terminos_encontrados_por_emocion

# --- Ejemplos de Uso ---
if __name__ == "__main__":
    print("--- Probando el algoritmo de detección de emociones ---")

    # Ejemplos de entrada de usuario
    ejemplos_texto = [
        "Me siento muy agüitado hoy. No sé qué hacer con esta tristeza.",
        "¡Qué chido! La vida es a toda madre y me siento muy feliz.",
        "Estoy bien enojado, me sacó de onda lo que pasó. ¡Me hierve la sangre!",
        "Ando con el Jesús en la boca, me paniqueé con el susto. No siento miedo.", # "No siento miedo" para probar negación
        "Estoy tranquilo, no siento ninguna emoción fuerte.",
        "Me da cosa ver eso, qué oso tan grande.",
        "Tengo un cargo de conciencia horrible.", # Prueba de nueva emoción
        "No me siento nada bien, estoy un poco deprimido.",
        "Esto es fabuloso y me hace sentir contento.",
        "No estoy enojado, solo un poco irritado.",
        "El día está hermoso, pero no me siento triste.",
        "Tengo mucha ansiedad y nerviosismo.",
        "Esto es una fobia terrible.",
        "No tengo temor.",
        "Me siento súper mal, fatal, muy muy triste."
    ]

    for i, texto in enumerate(ejemplos_texto):
        print(f"\n--- Analizando ejemplo {i+1}: '{texto}' ---")
        emocion, conteos, terminos_encontrados = detectar_emocion(texto, diccionario_emociones_df)

        print(f"\nRESULTADO FINAL:")
        print(f"  Emoción principal detectada: {emocion}")
        print(f"  Conteo de emociones: {dict(conteos)}")
        print(f"  Términos que contribuyeron: {dict(terminos_encontrados)}")

       
        if emocion == "NO_DETECTADA" or emocion == "EMPATE":
            print("  -> (Respuesta del Bot): No pude identificar tu emoción con claridad. ¿Puedes elegir una de las opciones?")
        else:
            
            # Primero, intenta encontrar la primera palabra clave detectada
            palabra_ejemplo_detectada = None
            if emocion in terminos_encontrados and terminos_encontrados[emocion]:
                palabra_ejemplo_detectada = terminos_encontrados[emocion][0] # Toma el primer término que contribuyó
            
            respuesta_en_csv = f"Entiendo que te sientes {emocion}." # Respuesta por defecto

            if palabra_ejemplo_detectada:
                # Busca el término en el diccionario para obtener la respuesta asociada si existe
              
                matching_row = diccionario_emociones_df[
                    diccionario_emociones_df['Termino_emocional'].str.lower() == palabra_ejemplo_detectada.lower()
                ]
                if not matching_row.empty:
                    respuesta_en_csv = matching_row['Respuesta'].iloc[0] # Obtiene la respuesta de la fila
            
            print(f"  -> (Respuesta del Bot - Preliminar): {respuesta_en_csv}")