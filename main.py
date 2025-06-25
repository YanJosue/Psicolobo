import spacy
import pandas as pd
from fuzzywuzzy import fuzz
from collections import defaultdict


# --- 0. Cargar la Base de Conocimiento

def cargar_diccionario_emociones(ruta_csv='baseConocimientos/emociones_v2.csv', ruta_emolex_csv='baseConocimientos/emolex_es_estandarizado.csv'):
    """
    Carga el diccionario de emociones desde emociones_v2.csv y lo combina con el léxico EmoLex estandarizado.
    Retorna un DataFrame de pandas. Si hay errores críticos en la carga, retorna None.
    """
    df_actual = None
    df_emolex = None

    # Try loading the main dictionary
    try:
        df_actual = pd.read_csv(ruta_csv, encoding='utf-8', sep=',') #
        print(f"DEBUG: Nombres de columnas detectadas por pandas (emociones_v2): {df_actual.columns.tolist()}")
        print(f"Diccionario de emociones cargado exitosamente desde '{ruta_csv}'.")
    except FileNotFoundError:
        print(f"ERROR: Archivo principal de emociones '{ruta_csv}' NO ENCONTRADO. No se puede continuar.")
        return None # Critical error, return None
    except Exception as e:
        print(f"ERROR: Error al cargar el archivo principal de emociones '{ruta_csv}': {e}")
        print("Asegúrate de que el CSV esté bien formado y con la codificación correcta (UTF-8).")
        return None # Critical error, return None

    # Try loading the EmoLex dictionary
    try:
        df_emolex = pd.read_csv(ruta_emolex_csv, encoding='utf-8', sep=',') #
        print(f"DEBUG: Nombres de columnas detectadas por pandas (EmoLex estandarizado): {df_emolex.columns.tolist()}")
        print(f"Léxico EmoLex estandarizado cargado exitosamente desde '{ruta_emolex_csv}'.")
    except FileNotFoundError:
        print(f"Advertencia: No se encontró el archivo EmoLex estandarizado en '{ruta_emolex_csv}'. Continuando solo con emociones_v2.csv.")
        return df_actual # Return only the main DataFrame if EmoLex is not found
    except Exception as e:
        print(f"Advertencia: Error al cargar EmoLex estandarizado desde '{ruta_emolex_csv}': {e}. Continuando solo con emociones_v2.csv.")
        return df_actual # Return only the main DataFrame if EmoLex has issues


    if df_actual is not None and df_emolex is not None:
        df_actual['source_priority'] = 1 # Prioridad alta para tu léxico original
        df_emolex['source_priority'] = 2 # Prioridad más baja para EmoLex

        df_temp_combined = pd.concat([df_actual, df_emolex], ignore_index=True)

        # Ordenar para que las entradas de 'emociones_v2.csv' aparezcan primero
        df_temp_combined.sort_values(by='source_priority', inplace=True)

        # Eliminar duplicados, manteniendo la primera aparición (la de mayor prioridad)
        
        
        df_final = df_temp_combined.drop_duplicates(subset=['Termino_emocional'], keep='first') # Changed subset to only 'Termino_emocional'

        # Eliminar la columna temporal de prioridad
        df_final = df_final.drop(columns=['source_priority'])

        print("Diccionario de emociones y EmoLex estandarizado combinados exitosamente con priorización.")
        return df_final
    
    else:
        
        if df_actual is not None:
            return df_actual
        elif df_emolex is not None:
            return df_emolex
        else:
            return None 

# Cargar el diccionario al inicio

diccionario_emociones_df = cargar_diccionario_emociones(ruta_emolex_csv='baseConocimientos/emolex_es_estandarizado.csv') #


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
# Umbral difuso ajustado a 90 para menos falsos positivos
def detectar_emocion(texto_usuario: str, diccionario_df: pd.DataFrame, umbral_difuso: int = 90):
    """
    Implementa el algoritmo de detección de emociones basado en léxico de SERMO.
    Retorna la emoción detectada y el conteo de términos por emoción.
    """
    if diccionario_df is None: #
        print("Error interno: El diccionario de emociones no fue cargado en detectar_emocion.")
        return "ERROR_DIC_NO_CARGADO", {}, {}

    # 1. Sentence segmentation
    doc = nlp(texto_usuario) #
    oraciones = [sent.text for sent in doc.sents] #
    print(f"\nOraciones detectadas: {oraciones}")

    # Inicializar conteo de emociones
    conteo_emociones = defaultdict(int) #
    terminos_encontrados_por_emocion = defaultdict(list) #

    for oracion in oraciones: #
        doc_oracion = nlp(oracion)
        oracion_lower = oracion.lower() # Convertir la oración completa a minúsculas una vez

        matched_in_oracion = False # Bandera para saber si se encontró algo significativo en esta oración

        # --- Lógica de Detección de Frases (Multi-Word Expressions) ---
        # Prioridad alta para evitar coincidencias parciales con palabras individuales

        # Obtener todas las frases y sinónimos multi-palabra del diccionario
        dic_frases_raw = diccionario_df[diccionario_df['Termino_emocional'].str.contains(' ', na=False)]['Termino_emocional'].apply(lambda x: str(x).lower()).tolist()

        # Añadir sinónimos que sean frases
        for idx, row in diccionario_df.iterrows():
            if pd.notna(row['Lista_sinonimos']) and str(row['Lista_sinonimos']).strip(): #
                sinonimos = str(row['Lista_sinonimos']).lower().split(', ') #
                for s in sinonimos:
                    if ' ' in s:
                        dic_frases_raw.append(s)

        dic_frases = list(set(dic_frases_raw)) # Eliminar duplicados
        dic_frases.sort(key=len, reverse=True) # Ordenar de más largas a más cortas

        for frase_dic in dic_frases:
            if frase_dic in oracion_lower:

                # Verificar negación para la frase completa
                is_negated_phrase = False
                phrase_start_index = oracion_lower.find(frase_dic)
                if phrase_start_index > 0:
                    context_before_phrase = oracion_lower[max(0, phrase_start_index - 15):phrase_start_index].split() # Más contexto
                    if any(neg_word in context_before_phrase for neg_word in PALABRAS_NEGACION): #
                        is_negated_phrase = True

                # Buscar la fila correspondiente en el diccionario
                row_match = diccionario_df[
                    (diccionario_df['Termino_emocional'].str.lower() == frase_dic) |
                    (diccionario_df['Lista_sinonimos'].str.lower().str.contains(frase_dic, na=False))
                ]

                if not row_match.empty:
                    matched_row = row_match.iloc[0]
                    emocion = str(matched_row['Emocion']).lower() #
                    negacion_considerar_str = str(matched_row['Negacion_Considerar']).lower() #

                    if is_negated_phrase and negacion_considerar_str == 'sí':
                        print(f"  -> Frase '{frase_dic}' ignorada debido a negación.")
                        
                    else:
                        conteo_emociones[emocion] += 1 #
                        terminos_encontrados_por_emocion[emocion].append(frase_dic) #
                        print(f"  -> Coincidencia de FRASE: '{frase_dic}' -> {emocion}")
                        matched_in_oracion = True # Se encontró una coincidencia importante

                if matched_in_oracion: # Si se encontró una frase, no procesar palabras individuales para esta oración
                    break # Salir del bucle de frases para esta oración

        # Solo procesar tokens individuales si no se encontró una frase relevante en la oración
        if not matched_in_oracion:
            processed_tokens = [] #
            for token in doc_oracion: #
                # Incluye solo tokens alfabéticos (que también incluye caracteres con tilde, eñes)
                # Excluye stop words para reducir ruido, pero cuidado con frases idiomáticas.
                if token.is_alpha and not token.is_stop: #
                    processed_tokens.append({'original': token.text, 'lemma': token.lemma_.lower(), 'is_negated': False, 'spacy_token_obj': token}) #

            print(f"Tokens procesados (lemas) en la oración '{oracion}': {[t['lemma'] for t in processed_tokens]}")

            # --- Lógica de Negación (revisada) para tokens individuales ---
            for i, p_token in enumerate(processed_tokens):
                token_obj = p_token['spacy_token_obj']
                start_idx = max(0, token_obj.i - 3)
                contexto_previo_original = [t.text.lower() for t in doc_oracion[start_idx:token_obj.i]]

                if any(neg_word in contexto_previo_original for neg_word in PALABRAS_NEGACION):
                    p_token['is_negated'] = True
                    print(f"  -> '{p_token['original']}' (lema: '{p_token['lemma']}') marcado como negado debido a contexto: {contexto_previo_original}")

            # 5. Emotion recognition (Lexicon-based con Fuzzy Matching) para tokens individuales
            for p_token in processed_tokens:
                token_original_text = p_token['original']
                token_lemma = p_token['lemma']

                for idx, row in diccionario_df.iterrows():
                    termino_emocional_dic = str(row['Termino_emocional']).lower()
                    lista_sinonimos_dic = str(row['Lista_sinonimos']).lower().split(', ') #
                    emocion = str(row['Emocion']).lower()
                    negacion_considerar_str = str(row['Negacion_Considerar']).lower() #

                    if p_token['is_negated'] and negacion_considerar_str == 'sí':
                        continue #

                    # Prioridad 1: Coincidencia exacta con el término emocional del diccionario
                    if token_lemma == termino_emocional_dic:
                        conteo_emociones[emocion] += 1
                        terminos_encontrados_por_emocion[emocion].append(token_original_text)
                        print(f"  -> Coincidencia EXACTA (lema): '{token_lemma}' ('{token_original_text}') -> {emocion} (vs '{termino_emocional_dic}')")
                        break #

                    # Prioridad 2: Coincidencia exacta con algún sinónimo en la lista
                    if token_lemma in lista_sinonimos_dic:
                        conteo_emociones[emocion] += 1
                        terminos_encontrados_por_emocion[emocion].append(token_original_text)
                        print(f"  -> Coincidencia EXACTA (sinónimo): '{token_lemma}' ('{token_original_text}') -> {emocion} (en sinónimos de '{termino_emocional_dic}')")
                        break

                    # Prioridad 3: Coincidencia difusa con el término emocional (solo si no hubo coincidencia exacta)
                    similitud_termino = fuzz.ratio(token_lemma, termino_emocional_dic)
                    if similitud_termino >= umbral_difuso:
                        conteo_emociones[emocion] += 1
                        terminos_encontrados_por_emocion[emocion].append(token_original_text)
                        print(f"  -> Coincidencia DIFUSA ({similitud_termino}%): '{token_lemma}' ('{token_original_text}') con '{termino_emocional_dic}' -> {emocion}")
                        break #

                    # Prioridad 4: Coincidencia difusa con algún sinónimo
                    for sinonimo_dic in lista_sinonimos_dic:
                        if sinonimo_dic: # Check if synonym is not an empty string
                            similitud_sinonimo = fuzz.ratio(token_lemma, sinonimo_dic) #
                            if similitud_sinonimo >= umbral_difuso: #
                                conteo_emociones[emocion] += 1 #
                                terminos_encontrados_por_emocion[emocion].append(token_original_text) #
                                print(f"  -> Coincidencia DIFUSA (sinónimo, {similitud_sinonimo}%): '{token_lemma}' ('{token_original_text}') con sinónimo '{sinonimo_dic}' de '{termino_emocional_dic}' -> {emocion}") #
                                break #
                    else: #
                        continue #


    # 6. Voting
    if not conteo_emociones or all(v == 0 for v in conteo_emociones.values()):
        emocion_detectada = "NO_DETECTADA"
    else:
        emocion_detectada = max(conteo_emociones, key=conteo_emociones.get)
        max_conteo = conteo_emociones[emocion_detectada]
        # Verificar si hay empate
        if list(conteo_emociones.values()).count(max_conteo) > 1:
            emocion_detectada = "EMPATE" # O podrías devolver una lista de emociones empatadas

    # --- ADD THESE DEBUG PRINTS ---
    print(f"DEBUG: emocion_detectada type: {type(emocion_detectada)}, value: {emocion_detectada}")
    print(f"DEBUG: conteos type: {type(conteo_emociones)}, value: {dict(conteo_emociones)}")
    print(f"DEBUG: terminos_encontrados type: {type(terminos_encontrados_por_emocion)}, value: {dict(terminos_encontrados_por_emocion)}")
    # --- END DEBUG PRINTS ---

    return emocion_detectada, conteo_emociones, terminos_encontrados_por_emocion # Corrected return statement

# --- Ejemplos de Uso ---
if __name__ == "__main__":
    print("--- Probando el algoritmo de detección de emociones ---")

    # Ejemplos de entrada de usuario
    ejemplos_texto = [
        "Me siento muy agüitado hoy. No sé qué hacer con esta tristeza.",
        "Estoy bien enojado, me sacó de onda lo que pasó. ¡Me hierve la sangre!",
        "Ando con el Jesús en la boca, me paniqueé con el susto. No siento miedo.", # "No siento miedo" para probar negación
        "Tengo un cargo de conciencia horrible.", # Prueba de nueva emoción

    ]

    for i, texto in enumerate(ejemplos_texto):
        print(f"\n--- Analizando ejemplo {i+1}: '{texto}' ---")
        emocion, conteos, terminos_encontrados = detectar_emocion(texto, diccionario_emociones_df) #

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
                    # Asegúrate de que 'Respuesta' no esté vacía antes de usarla
                    if not pd.isna(matching_row['Respuesta'].iloc[0]) and matching_row['Respuesta'].iloc[0].strip() != '':
                        respuesta_en_csv = matching_row['Respuesta'].iloc[0] # Obtiene la respuesta de la fila

            print(f"  -> (Respuesta del Bot - Preliminar): {respuesta_en_csv}")
