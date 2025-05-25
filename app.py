import os
from flask import Flask, render_template, request, jsonify, send_file
import whisper
import tempfile
import subprocess
from datetime import timedelta
import zipfile
import shutil
from openai import OpenAI
import requests
import json
import re

# client = OpenAI()

# PALABRAS_CLAVE = ["creación", "trascendencia", "singularidad", "evolución", "dios", "universo", "programador", "perfecta", "cumpleaños", "feliz"]
PALABRAS_CLAVE = [
  "ay amor",
  "ayúdame con un video",
  "no te voy a grabar",
  "dime algo",
  "un chisme",
  "qué era",
  "se le escuchas que pendeja",
  "provense a heterotexas",
  "te vas queda como wachan",
  "lo demás menos de ahí",
  "la respondió a pesar la gorda",
  "preguntó si la gorda",
  "me dices tu vacía la graduación",
  "por qué estaría invita",
  "los chicos viven juntos",
  "has interactuado yo",
  "para pasarte la de las tiches",
  "le botó a su ahí",
  "ah y eso del cine",
  "cállate, caresca",
  "corto de presupuesto",
  "te debía",
  "me dijo un visto",
  "ella le bloqueó"
]


app = Flask(__name__)

# Cargar el modelo de Whisper
model = whisper.load_model("base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribir', methods=['POST'])
def transcribir_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo de audio'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400

        # Validar que el archivo es de un formato aceptado
        allowed_extensions = ['.mp3', '.wav', '.mp4', '.opus', '.m4a']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': 'El archivo debe ser en formato MP3, WAV o MP4'}), 400

        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_input:
            file.save(temp_input.name)
            input_path = temp_input.name

        # Convertir el archivo a formato WAV usando FFmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            output_path = temp_output.name

        # Ejecutar FFmpeg para convertir el archivo a WAV mono de 16kHz
        command = [
            'ffmpeg', '-i', input_path,
            '-ar', '16000', '-ac', '1',
            '-c:a', 'pcm_s16le', output_path,
            '-y'  # Sobrescribir si el archivo existe
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Eliminar el archivo de entrada temporal
        os.remove(input_path)

        # Obtener el idioma seleccionado
        language = request.form.get('language')

        # Realizar la transcripción con Whisper
        result = model.transcribe(output_path, language=language)

        # Eliminar el archivo de audio convertido temporal
        os.remove(output_path)

        # Guardar la transcripción en archivos de texto y SRT temporalmente
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w', encoding='utf-8') as temp_transcript_txt:
            temp_transcript_txt.write(result['text'])
            transcript_txt_path = temp_transcript_txt.name

        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False, mode='w', encoding='utf-8') as temp_transcript_srt:
            srt_content = generate_srt(result['segments'])
            temp_transcript_srt.write(srt_content)
            transcript_srt_path = temp_transcript_srt.name

        # Preparar la respuesta para descargar los archivos
        response_format = request.form.get('response_format')

        if response_format == 'srt':
            return send_file(transcript_srt_path, as_attachment=True, download_name="transcripcion.srt")
        else:
            return send_file(transcript_txt_path, as_attachment=True, download_name="transcripcion.txt")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_srt(segments):
    srt_content = ""
    for i, segment in enumerate(segments):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        srt_content += f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

@app.route('/generar-clips', methods=['POST'])
def generar_clips_virales():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo de video'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400

        allowed_extensions = ['.mp4', '.mp3', '.wav']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': 'El archivo debe ser MP4, MP3 o WAV'}), 400

        # Guardar archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_input:
            file.save(temp_input.name)
            input_path = temp_input.name

        # Convertir a WAV mono 16kHz
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name

        ffmpeg_command = [
            "ffmpeg", "-i", input_path,
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", audio_path,
            "-y"
        ]
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception("Error al convertir el archivo de audio:\n" + result.stderr.decode())

        # Transcripción con segmentos
        result = model.transcribe(audio_path, word_timestamps=False)
        segmentos = result.get('segments', [])
        momentos = detectar_momentos_virales_con_ia(segmentos)
        
        # # Agrupar por bloques de 30 segundos
        # grupos = agrupar_segmentos(segmentos, duracion_maxima=30.0)

        # # Detectar momentos virales en grupos largos
        # momentos = detectar_momentos_virales_grandes(grupos)

        # momentos = detectar_momentos_virales(segmentos)

        if not momentos:
            return jsonify({'message': 'No se encontraron momentos virales'}), 200

        # Carpeta temporal para clips
        clips_folder = tempfile.mkdtemp()
        clip_paths = []

        for i, momento in enumerate(momentos):
            clip_path = os.path.join(clips_folder, f"clip_{i+1}.mp4")
            crear_clip(input_path, momento['start'], momento['end'], clip_path)
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clip_paths.append(clip_path)
            else:
                print(f"[WARN] No se creó el clip: {clip_path}")

        if not clip_paths:
            return jsonify({'error': 'Se encontraron momentos virales, pero no se generaron clips válidos'}), 500

        # Comprimir en ZIP
        zip_path = os.path.join(tempfile.gettempdir(), "clips_virales.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for clip in clip_paths:
                zipf.write(clip, os.path.basename(clip))

        # Limpieza
        os.remove(input_path)
        os.remove(audio_path)
        shutil.rmtree(clips_folder)

        return send_file(zip_path, as_attachment=True, download_name="clips_virales.zip")

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# def detectar_momentos_virales(segments):
#     momentos = []
#     for seg in segments:
#         texto = seg['text'].lower()
#         if any(palabra in texto for palabra in PALABRAS_CLAVE):
#             momentos.append({
#                 'start': seg['start'],
#                 'end': seg['end'],
#                 'text': seg['text']
#             })
#     return momentos
# def detectar_momentos_virales(segments, min_duracion=3.0, max_gap=1.0):
#     momentos = []
#     current_momento = None

#     for seg in segments:
#         texto = seg['text'].lower()
#         if any(palabra in texto for palabra in PALABRAS_CLAVE):
#             if current_momento is None:
#                 current_momento = {
#                     'start': seg['start'],
#                     'end': seg['end'],
#                     'texts': [seg['text']]
#                 }
#             else:
#                 # Si el segmento actual está muy cerca del anterior, lo unimos
#                 if seg['start'] - current_momento['end'] <= max_gap:
#                     current_momento['end'] = seg['end']
#                     current_momento['texts'].append(seg['text'])
#                 else:
#                     # Guardamos momento anterior si dura suficiente
#                     duracion = current_momento['end'] - current_momento['start']
#                     if duracion >= min_duracion:
#                         momentos.append({
#                             'start': current_momento['start'],
#                             'end': current_momento['end'],
#                             'text': " ".join(current_momento['texts'])
#                         })
#                     # Empezamos nuevo momento
#                     current_momento = {
#                         'start': seg['start'],
#                         'end': seg['end'],
#                         'texts': [seg['text']]
#                     }
#     # Agregar el último momento si aplica
#     if current_momento:
#         duracion = current_momento['end'] - current_momento['start']
#         if duracion >= min_duracion:
#             momentos.append({
#                 'start': current_momento['start'],
#                 'end': current_momento['end'],
#                 'text': " ".join(current_momento['texts'])
#             })

#     return momentos
def agrupar_segmentos(segments, duracion_maxima=30.0):
    grupos = []
    grupo_actual = []
    tiempo_inicial = None

    for seg in segments:
        if not grupo_actual:
            tiempo_inicial = seg['start']

        grupo_actual.append(seg)
        duracion = seg['end'] - tiempo_inicial

        if duracion >= duracion_maxima:
            grupos.append(grupo_actual)
            grupo_actual = []

    if grupo_actual:
        grupos.append(grupo_actual)

    return grupos

def detectar_momentos_virales_grandes(grupos):
    momentos = []

    for grupo in grupos:
        texto_completo = " ".join(seg['text'].strip() for seg in grupo)

        if es_momento_viral(texto_completo):
            momentos.append({
                'start': grupo[0]['start'],
                'end': grupo[-1]['end'],
                'text': texto_completo
            })

    return momentos



def detectar_momentos_virales(segments):
    momentos = []

    for seg in segments:
        texto = seg['text'].strip()

        # Opción 1: mantener la detección por palabras clave (opcional)
        contiene_palabras_clave = any(palabra in texto.lower() for palabra in PALABRAS_CLAVE)

        # Opción 2: análisis semántico con GPT
        if es_momento_viral(texto):
            momentos.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': texto
            })

    return momentos



def crear_clip(input_path, start, end, output_path):
    duration = end - start
    command = [
        'ffmpeg', '-ss', str(start), '-i', input_path,
        '-t', str(duration),
        '-c:v', 'libx264', '-c:a', 'aac',
        output_path, '-y'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"[ERROR] al crear clip {output_path}:\n{result.stderr.decode()}")

def es_momento_viral(texto):
    prompt = f"""Analiza el siguiente texto transcrito de un video. 
    Dime si es un momento impactante, viral, motivador o divertido. Responde SOLO 'Sí' o 'No'.

    Texto: "{texto}"
    ¿Es viral?"""

    response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    print(response.json())
    return "sí" in response.json()["response"].lower()

def detectar_momentos_virales_con_ia(segmentos):
    # Construir transcripción con marcas de tiempo
    transcripcion_completa = ""
    for seg in segmentos:
        transcripcion_completa += f"[{seg['start']:.2f} --> {seg['end']:.2f}] {seg['text']}\n"

    prompt = f"""
Eres un asistente que analiza transcripciones de videos para detectar y construir momentos virales, impactantes, motivadores, graciosos o de interés general.

Te entrego una transcripción con marcas de tiempo. Cada fragmento es muy corto. Tu tarea es combinar **fragmentos consecutivos** en bloques más largos que tengan sentido juntos.

⚠️ REGLAS OBLIGATORIAS:

- Cada bloque viral debe tener una duración mínima de **10 segundos** (ideal: entre 15 y 60 segundos).
- Para lograrlo, **debes unir** varios fragmentos cortos que estén seguidos y relacionados o que se puedan entender como parte de un mismo tema.
- El texto del bloque será la suma de todos los textos de los fragmentos unidos (concatenados con espacio).
- El tiempo de inicio será el del primer fragmento unido.
- El tiempo de fin será el del último fragmento unido.
- Devuelve solo bloques virales largos, **NO devuelvas fragmentos individuales menores a 10 segundos**.
- Si no hay bloques válidos, devuelve una lista vacía: []

⚠️ IMPORTANTE:
- NO EXPLIQUES NADA.
- NO ESCRIBAS texto antes o después del JSON.
- SOLO devuelve una lista JSON válida.

Ejemplo de salida:

[
  {{
    "start": 10.0,
    "end": 28.7,
    "text": "Este fue el momento más impactante del video. Luego dijo algo que cambió todo. Y terminó con una frase épica."
  }},
  {{
    "start": 42.3,
    "end": 65.5,
    "text": "Una secuencia llena de humor que la audiencia adoró."
  }}
]

Ahora analiza esta transcripción y devuelve SOLO el JSON con los bloques virales combinados y con al menos 10 segundos de duración:

{transcripcion_completa}
"""





    # Llamada a OpenAI
    response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    print(response.json().get("response", ""))
    contenido = response.json().get("response", "")

    # Intenta extraer solo el JSON de la respuesta
    try:
        # Usa regex para detectar el JSON dentro del texto
        match = re.search(r"\[\s*{.*?}\s*\]", contenido, re.DOTALL)
        if match:
            json_puro = match.group(0)
            return json.loads(json_puro)
        else:
            print("No se encontró un JSON en la respuesta:")
            print(contenido)
            return []
    except json.JSONDecodeError as e:
        print("Error al interpretar el JSON:")
        print("Contenido crudo:", repr(contenido))
        print("Error:", str(e))
        return []



if __name__ == "__main__":
    app.run(debug=True)

