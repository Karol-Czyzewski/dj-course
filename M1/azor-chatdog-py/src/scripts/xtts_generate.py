#!/usr/bin/env python
"""Pomocniczy skrypt do generowania mowy XTTS w izolowanym interpreterze Python 3.12.x.

Użycie (wywoływany przez audio.py):
  python xtts_generate.py --text "Ala ma kota" --out /sciezka/plik.wav [--language pl] [--speaker-wav voice.wav] [--model <model_name>]

Wymagania: zainstalowany pakiet coqui-tts w tym interpreterze.
"""
import argparse
import sys

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--language', default='pl')
    ap.add_argument('--speaker-wav')
    ap.add_argument('--model', default='tts_models/multilingual/multi-dataset/xtts_v2')
    return ap.parse_args()

def main():
    args = parse_args()
    try:
        from TTS.api import TTS
    except ImportError:
        print('ERROR: coqui-tts nie jest zainstalowane w tym interpreterze.', file=sys.stderr)
        return 1
    try:
        tts = TTS(args.model, progress_bar=False).to('cpu')
        tts.tts_to_file(
            text=args.text,
            file_path=args.out,
            speaker_wav=args.speaker_wav if args.speaker_wav else None,
            language=args.language
        )
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())