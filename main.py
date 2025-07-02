import spacy
import pandas as pd
from fuzzywuzzy import fuzz
from collections import defaultdict
import os

# --- 0. Cargar la Base de Conocimiento

def cargar_diccionario_emociones(ruta_csv='baseConocimientos/base_conocimientos.csv'):
    """
    Carga el diccionario de emociones desde un único archivo CSV simple
    con las columnas "Entidad" y "Termino", y lo adapta al formato requerido.
    """
    print(f"Iniciando la carga de la base de conocimientos simple desde: '{ruta_csv}'")
    try:
        df_simple = pd.read_csv(ruta_csv, encoding='utf-8', sep=',')
        print(f"-> Archivo '{ruta_csv}' cargado exitosamente.")
        print(f"DEBUG: Columnas detectadas en el CSV: {df_simple.columns.tolist()}")

        df_completo = pd.DataFrame()

        df_completo['Emocion'] = df_simple['Entidad']
        df_completo['Termino_emocional'] = df_simple['Termino']
        df_completo['Lista_sinonimos'] = ''
        df_completo['Negacion_Considerar'] = 'sí'

        print("-> Base de conocimientos adaptada al formato interno del bot.")
        return df_completo

    except FileNotFoundError:
        print(f"ERROR: Archivo '{ruta_csv}' NO ENCONTRADO. Asegúrate de que el nombre sea correcto y esté en la misma carpeta.")
        return None
    except KeyError as e:
        print(f"ERROR: Falta una columna esperada en tu CSV. Asegúrate de que las columnas se llamen 'Entidad' y 'Termino'. Error: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Error al cargar o procesar el archivo '{ruta_csv}': {e}")
        return None

# --- NUEVA FUNCIÓN PARA LEER ARCHIVOS TXT ---
def leer_letra_cancion(ruta_archivo):
    """
    Lee un archivo de texto y devuelve su contenido como una cadena.
    """
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: El archivo '{ruta_archivo}' no fue encontrado.")
        return None
    except Exception as e:
        print(f"ERROR: Ocurrió un error al leer el archivo '{ruta_archivo}': {e}")
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
def detectar_emocion(texto_usuario: str, diccionario_df: pd.DataFrame, umbral_difuso: int = 90):
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
        oracion_lower = oracion.lower()

        matched_in_oracion = False

        # --- Lógica de Detección de Frases (Multi-Word Expressions) ---
        dic_frases_raw = diccionario_df[diccionario_df['Termino_emocional'].str.contains(' ', na=False)]['Termino_emocional'].apply(lambda x: str(x).lower()).tolist()

        for idx, row in diccionario_df.iterrows():
            if pd.notna(row['Lista_sinonimos']) and str(row['Lista_sinonimos']).strip():
                sinonimos = str(row['Lista_sinonimos']).lower().split(', ')
                for s in sinonimos:
                    if ' ' in s:
                        dic_frases_raw.append(s)

        dic_frases = list(set(dic_frases_raw))
        dic_frases.sort(key=len, reverse=True)

        for frase_dic in dic_frases:
            if frase_dic in oracion_lower:
                is_negated_phrase = False
                phrase_start_index = oracion_lower.find(frase_dic)
                if phrase_start_index > 0:
                    context_before_phrase = oracion_lower[max(0, phrase_start_index - 15):phrase_start_index].split()
                    if any(neg_word in context_before_phrase for neg_word in PALABRAS_NEGACION):
                        is_negated_phrase = True

                row_match = diccionario_df[
                    (diccionario_df['Termino_emocional'].str.lower() == frase_dic) |
                    (diccionario_df['Lista_sinonimos'].str.lower().str.contains(frase_dic, na=False))
                ]

                if not row_match.empty:
                    matched_row = row_match.iloc[0]
                    emocion = str(matched_row['Emocion']).lower()
                    negacion_considerar_str = str(matched_row['Negacion_Considerar']).lower()

                    if is_negated_phrase and negacion_considerar_str == 'sí':
                        print(f"  -> Frase '{frase_dic}' ignorada debido a negación.")
                    else:
                        conteo_emociones[emocion] += 1
                        terminos_encontrados_por_emocion[emocion].append(frase_dic)
                        print(f"  -> Coincidencia de FRASE: '{frase_dic}' -> {emocion}")
                        matched_in_oracion = True

                if matched_in_oracion:
                    break

        if not matched_in_oracion:
            processed_tokens = []
            for token in doc_oracion:
                if token.is_alpha and not token.is_stop:
                    processed_tokens.append({'original': token.text, 'lemma': token.lemma_.lower(), 'is_negated': False, 'spacy_token_obj': token})

            print(f"Tokens procesados (lemas) en la oración '{oracion}': {[t['lemma'] for t in processed_tokens]}")

            for i, p_token in enumerate(processed_tokens):
                token_obj = p_token['spacy_token_obj']
                start_idx = max(0, token_obj.i - 3)
                contexto_previo_original = [t.text.lower() for t in doc_oracion[start_idx:token_obj.i]]

                if any(neg_word in contexto_previo_original for neg_word in PALABRAS_NEGACION):
                    p_token['is_negated'] = True
                    print(f"  -> '{p_token['original']}' (lema: '{p_token['lemma']}') marcado como negado debido a contexto: {contexto_previo_original}")

            for p_token in processed_tokens:
                token_original_text = p_token['original']
                token_lemma = p_token['lemma']

                match_encontrado_para_token = False
                for idx, row in diccionario_df.iterrows():
                    termino_emocional_dic = str(row['Termino_emocional']).lower()
                    
                    lista_sinonimos_str = str(row['Lista_sinonimos']) if pd.notna(row['Lista_sinonimos']) else ""
                    lista_sinonimos_dic = lista_sinonimos_str.lower().split(', ')
                    
                    emocion = str(row['Emocion']).lower()
                    negacion_considerar_str = str(row['Negacion_Considerar']).lower()

                    if p_token['is_negated'] and negacion_considerar_str == 'sí':
                        continue

                    if token_lemma == termino_emocional_dic:
                        conteo_emociones[emocion] += 1
                        terminos_encontrados_por_emocion[emocion].append(token_original_text)
                        print(f"  -> Coincidencia EXACTA (lema): '{token_lemma}' ('{token_original_text}') -> {emocion} (vs '{termino_emocional_dic}')")
                        match_encontrado_para_token = True
                        break

                    if token_lemma in lista_sinonimos_dic:
                        conteo_emociones[emocion] += 1
                        terminos_encontrados_por_emocion[emocion].append(token_original_text)
                        print(f"  -> Coincidencia EXACTA (sinónimo): '{token_lemma}' ('{token_original_text}') -> {emocion} (en sinónimos de '{termino_emocional_dic}')")
                        match_encontrado_para_token = True
                        break

                    similitud_termino = fuzz.ratio(token_lemma, termino_emocional_dic)
                    if similitud_termino >= umbral_difuso:
                        conteo_emociones[emocion] += 1
                        terminos_encontrados_por_emocion[emocion].append(token_original_text)
                        print(f"  -> Coincidencia DIFUSA ({similitud_termino}%): '{token_lemma}' ('{token_original_text}') con '{termino_emocional_dic}' -> {emocion}")
                        match_encontrado_para_token = True
                        break

                    for sinonimo_dic in lista_sinonimos_dic:
                        if sinonimo_dic:
                            similitud_sinonimo = fuzz.ratio(token_lemma, sinonimo_dic)
                            if similitud_sinonimo >= umbral_difuso:
                                conteo_emociones[emocion] += 1
                                terminos_encontrados_por_emocion[emocion].append(token_original_text)
                                print(f"  -> Coincidencia DIFUSA (sinónimo, {similitud_sinonimo}%): '{token_lemma}' ('{token_original_text}') con sinónimo '{sinonimo_dic}' de '{termino_emocional_dic}' -> {emocion}")
                                match_encontrado_para_token = True
                                break
                    if match_encontrado_para_token:
                        break


    if not conteo_emociones or all(v == 0 for v in conteo_emociones.values()):
        emocion_detectada = "NO_DETECTADA"
    else:
        emocion_detectada = max(conteo_emociones, key=conteo_emociones.get)
        max_conteo = conteo_emociones[emocion_detectada]
        
        if list(conteo_emociones.values()).count(max_conteo) > 1:
            emocion_detectada = "EMPATE"

    print(f"DEBUG: emocion_detectada type: {type(emocion_detectada)}, value: {emocion_detectada}")
    print(f"DEBUG: conteos type: {type(conteo_emociones)}, value: {dict(conteo_emociones)}")
    print(f"DEBUG: terminos_encontrados type: {type(terminos_encontrados_por_emocion)}, value: {dict(terminos_encontrados_por_emocion)}")

    return emocion_detectada, conteo_emociones, terminos_encontrados_por_emocion

# --- Ejemplos de Uso ---
if __name__ == "__main__":
    print("--- Probando el algoritmo de detección de emociones en letras de canciones ---")
    
    carpeta_letras = "canciones"

    canciones_a_analizar = [
        "el_triste.txt",
        # "otra_cancion.txt" 
    ]

    for nombre_archivo in canciones_a_analizar:
        ruta_completa = os.path.join(carpeta_letras, nombre_archivo)
        
        print(f"\n--- Analizando canción: '{nombre_archivo}' ---")
        
        letra = leer_letra_cancion(ruta_completa)

        if letra:
            emocion, conteos, terminos_encontrados = detectar_emocion(letra, diccionario_emociones_df)

            print(f"\nRESULTADO FINAL PARA '{nombre_archivo}':")
            print(f"  Emoción principal detectada: {emocion}")
            print(f"  Conteo de emociones: {dict(conteos)}")
            print(f"  Términos que contribuyeron: {dict(terminos_encontrados)}")