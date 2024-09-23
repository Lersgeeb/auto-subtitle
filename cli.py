import argparse
import os
import sys
import subprocess
import whisper
import ffmpeg
import time

def create_assets_directory():
    assets_dir = os.path.join(os.getcwd(), 'assets')
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    return assets_dir

def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        print(f"El archivo {file_path} no existe.")
        sys.exit(1)

def has_audio_stream(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        streams = probe.get('streams', [])
        for stream in streams:
            if stream.get('codec_type') == 'audio':
                return True
        return False
    except ffmpeg.Error as e:
        print(f"Error al analizar el archivo {file_path}: {e}")
        sys.exit(1)

# Función personalizada para escribir el archivo SRT
def write_srt(segments, file):
    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip().replace('-->', '→')  # Reemplazar '-->' para evitar conflictos en el formato SRT
        file.write(f"{i}\n")
        file.write(f"{start} --> {end}\n")
        file.write(f"{text}\n\n")

def extract_subtitles(input_file, model_name, language, task, subtitle_output_path, verbose):
    if not has_audio_stream(input_file):
        print(f"El archivo {input_file} no contiene una pista de audio. No se pueden extraer subtítulos.")
        return None

    model = whisper.load_model(model_name)
    if verbose:
        print(f"Cargando el modelo Whisper '{model_name}'...")
        print("Iniciando la transcripción...")

    result = model.transcribe(input_file, language=language, task=task, verbose=verbose)

    if not os.path.exists(subtitle_output_path):
        os.makedirs(subtitle_output_path)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    srt_file = os.path.join(subtitle_output_path, f"{base_name}.srt")

    with open(srt_file, 'w', encoding='utf-8') as f:
        write_srt(result['segments'], f)  # Utilizar la función personalizada write_srt

    if verbose:
        print(f"Subtítulos extraídos y guardados en {srt_file}")

    return srt_file  # Devolver la ruta del archivo SRT generado

def transform_audio_to_video_with_subtitles(input_file, srt_file, output_path, verbose):
    if not has_audio_stream(input_file):
        print(f"El archivo {input_file} no contiene una pista de audio. No se puede transformar audio en video.")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_path, f"{base_name}_subtitled.mp4")

    if verbose:
        print(f"Transformando audio a video con visualización y agregando subtítulos...")

    # Simulación de progreso
    total_steps = 100
    for i in range(total_steps):
        time.sleep(0.01)  # Simula el tiempo de procesamiento
        percent_complete = (i + 1) / total_steps * 100
        print(f"Progreso: {percent_complete:.0f}% completado", end='\r')

    try:
        # Cargar el archivo de entrada
        input_stream = ffmpeg.input(input_file)

        # Crear el flujo de video a partir del audio utilizando el filtro showwaves
        video_stream = (
            input_stream.audio
            .filter('showwaves', s='1280x720', mode='cline', colors='white')
            .setpts('PTS-STARTPTS')
        )

        # Agregar subtítulos al video
        video_stream = video_stream.filter('subtitles', srt_file)

        # Crear el flujo de salida combinando el video y el audio original
        ffmpeg_output = ffmpeg.output(
            video_stream,
            input_stream.audio,
            output_file,
            pix_fmt='yuv420p',
            vcodec='libx264',
            acodec='aac',
            audio_bitrate='192k',
            format='mp4',
            strict='experimental'
        )

        if not verbose:
            ffmpeg_output = ffmpeg_output.global_args('-hide_banner', '-loglevel', 'error')

        ffmpeg_output.overwrite_output().run()
    except ffmpeg.Error as e:
        print(f"\nError al generar el video: {e.stderr.decode()}")
        sys.exit(1)

    if verbose:
        print(f"\nVideo generado en {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Procesamiento de audio y video",
        epilog="""
Ejemplos de uso:

- Extracción de subtítulos:
  python script.py "video.mp4" --extract_subtitles --subtitle_output_path "./assets/subtitles"

- Transformación de audio en video con visualización:
  python script.py "audio.mp3" --audio_to_video -o "./assets/videos"

- Transformación de audio a video con subtítulos embebidos:
  python script.py "audio.mp3" --audio_to_video_with_subtitles --task translate --language es -o "./assets/videos"

- Combinación de funcionalidades:
  python script.py "audio.mp3" --combine --subtitle_output_path "./assets/subtitles" -o "./assets/videos"
        """
    )
    parser.add_argument('input_file', type=str, help='Archivo de entrada de audio o video')
    parser.add_argument('-o', '--output_path', type=str, default='./assets', help='Ruta de salida para archivos generados')
    parser.add_argument('--extract_subtitles', action='store_true', help='Extraer subtítulos del archivo de entrada')
    parser.add_argument('--audio_to_video', action='store_true', help='Transformar audio en video con visualización')
    parser.add_argument('--audio_to_video_with_subtitles', action='store_true', help='Transformar audio en video con visualización y subtítulos embebidos')
    parser.add_argument('--combine', action='store_true', help='Combinar extracción de subtítulos y transformación de audio a video')
    parser.add_argument('--subtitle_output_path', type=str, default='./assets/subtitles', help='Ruta para guardar los subtítulos')
    parser.add_argument('--model', type=str, default='small', help='Modelo Whisper a utilizar')
    parser.add_argument('--language', type=str, default=None, help='Idioma del audio o idioma de destino para traducción')
    parser.add_argument('--task', type=str, choices=['transcribe', 'translate'], default='transcribe', help='Tarea a realizar por Whisper')
    parser.add_argument('--verbose', action='store_true', help='Mostrar mensajes detallados durante la ejecución')

    args = parser.parse_args()

    assets_dir = create_assets_directory()
    check_file_exists(args.input_file)

    srt_file = None

    if args.extract_subtitles or args.combine or args.audio_to_video_with_subtitles:
        if args.verbose:
            print("Iniciando extracción de subtítulos...")
        srt_file = extract_subtitles(
            input_file=args.input_file,
            model_name=args.model,
            language=args.language,
            task=args.task,
            subtitle_output_path=args.subtitle_output_path,
            verbose=args.verbose
        )
        if args.verbose:
            print("Extracción de subtítulos completada.")

    if args.audio_to_video or args.combine:
        if args.verbose:
            print("Iniciando transformación de audio a video...")
        transform_audio_to_video(
            input_file=args.input_file,
            output_path=args.output_path,
            verbose=args.verbose
        )
        if args.verbose:
            print("Transformación de audio a video completada.")

    if args.audio_to_video_with_subtitles:
        if srt_file is None:
            print("No se pudieron generar subtítulos. Abortando.")
            sys.exit(1)
        if args.verbose:
            print("Iniciando transformación de audio a video con subtítulos...")
        transform_audio_to_video_with_subtitles(
            input_file=args.input_file,
            srt_file=srt_file,
            output_path=args.output_path,
            verbose=args.verbose
        )
        if args.verbose:
            print("Transformación de audio a video con subtítulos completada.")

if __name__ == "__main__":
    main()
