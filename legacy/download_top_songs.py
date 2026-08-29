#!/usr/bin/env python3
"""
Downloads top 20 richsync songs & TTML from Unison (Better Lyrics) into temp/songs/
"""

import os
import re
import sys
import json
import urllib.request
import urllib.parse
import subprocess

TARGET_SONGS = [
    {"videoId": "R-hYM3BqTbA", "name": "Self Aware", "artist": "Temper City"},
    {"videoId": "ZVgHPSyEIqk", "name": "Let Down", "artist": "Radiohead"},
    {"videoId": "T0Ry1m0Blug", "name": "Harvey", "artist": "Her's"},
    {"videoId": "oE56g61mW44", "name": "Isn't She Lovely", "artist": "Stevie Wonder"},
    {"videoId": "HkSUnEiSVYM", "name": "Charlie's Inferno", "artist": "That Handsome Devil"},
    {"videoId": "kbywKmQsz7I", "name": "Catch Me If You Can", "artist": "Alan Walker"},
    {"videoId": "9OUurVdRGsc", "name": "Forever Young", "artist": "Alphaville"},
    {"videoId": "Iqw03oGysxE", "name": "Run Rabbit", "artist": "Mollie Elizabeth"},
    {"videoId": "Wzn4BLtE73o", "name": "Who Knows", "artist": "Daniel Caesar"},
    {"videoId": "vXc5jYyfRqY", "name": "One More Hour", "artist": "Tame Impala"},
    {"videoId": "_t1vqLwqbyA", "name": "Be Quiet and Drive (Far Away)", "artist": "Deftones"},
    {"videoId": "6fVlX2AbW_U", "name": "A Thousand Years", "artist": "John Michael Howell"},
    {"videoId": "uxyM7vhU0uU", "name": "Dream Sweet in Sea Major", "artist": "Miracle Musical"},
    {"videoId": "vOfyxPRunFM", "name": "Animal", "artist": "KATSEYE"},
    {"videoId": "unypqeWmyNI", "name": "FATHER", "artist": "Kanye West"},
    {"videoId": "K_8yRH2KPVo", "name": "Superman", "artist": "Eminem"},
    {"videoId": "k0g04t7ZeSw", "name": "怪物 (Monster)", "artist": "YOASOBI"},
    {"videoId": "B95OUKk7alM", "name": "Touch The Sky", "artist": "Kanye West"},
    {"videoId": "WWa28QJEjnQ", "name": "the perfect pair", "artist": "beabadoobee"},
    {"videoId": "W0gRQYSU5cw", "name": "It Doesn't Matter", "artist": "The Living Tombstone"},
    {"videoId": "qwgyPtMmNK4", "name": "Void", "artist": "Jim Yosef"},
    {"videoId": "7n1pMfm-n8A", "name": "Eenie Meenie", "artist": "Sean Kingston & Justin Bieber"},
    {"videoId": "F_X3qKVZrWU", "name": "From the Start", "artist": "Good Kid"},
]

def sanitize_filename(name):
    clean = re.sub(r'[/\\?%*:|"<>#]', '', name).strip()
    return clean

def main():
    songs_dir = os.path.join(os.path.dirname(__file__), 'songs')
    os.makedirs(songs_dir, exist_ok=True)

    print("=" * 70)
    print("Downloading Top 20+ Unison Better Lyrics & Audio Tracks")
    print("=" * 70)

    downloaded = 0
    for idx, item in enumerate(TARGET_SONGS, 1):
        vid = item["videoId"]
        raw_name = f"{item['name']} - {item['artist']}"
        safe_name = sanitize_filename(raw_name)
        
        txt_path = os.path.join(songs_dir, f"{safe_name}.txt")
        mp3_path = os.path.join(songs_dir, f"{safe_name}.mp3")

        print(f"\n[{idx:2d}/{len(TARGET_SONGS)}] {raw_name} (ID: {vid})")

        # 1. Fetch TTML lyrics
        if not os.path.exists(txt_path):
            try:
                url = f"https://unison.boidu.dev/lyrics?v={vid}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    if data.get('success') and 'lyrics' in data.get('data', {}):
                        ttml = data['data']['lyrics']
                        with open(txt_path, 'w', encoding='utf-8') as fh:
                            fh.write(ttml)
                        print(f"  Saved lyrics: {os.path.basename(txt_path)}")
                    else:
                        print("  No TTML lyrics found in API response, skipping.")
                        continue
            except Exception as e:
                print(f"  Error fetching lyrics: {e}")
                continue
        else:
            print(f"  Lyrics already present: {os.path.basename(txt_path)}")

        # 2. Download audio stream via yt-dlp
        if not os.path.exists(mp3_path):
            yt_url = f"https://www.youtube.com/watch?v={vid}"
            temp_template = os.path.join(songs_dir, f"{safe_name}.%(ext)s")
            cmd = [
                'yt-dlp',
                '-x',
                '--audio-format', 'mp3',
                '--audio-quality', '128K',
                '-o', temp_template,
                '--no-playlist',
                '--quiet',
                yt_url
            ]
            try:
                print("  Downloading audio stream...")
                subprocess.run(cmd, check=True, timeout=60)
                if os.path.exists(mp3_path):
                    print(f"  Downloaded MP3: {os.path.basename(mp3_path)}")
                    downloaded += 1
                else:
                    print("  Audio extraction finished with non-mp3 extension.")
            except Exception as e:
                print(f"  Error downloading audio: {e}")
        else:
            print(f"  MP3 already present: {os.path.basename(mp3_path)}")
            downloaded += 1

    print("\n" + "=" * 70)
    print(f"Finished! Total songs active in temp/songs/: {len([f for f in os.listdir(songs_dir) if f.endswith('.txt')])}")

    # Invalidate cache so optimizer re-indexes full dataset
    cache_file = os.path.join(os.path.dirname(__file__), 'cache', 'song_dataset_cache.pkl')
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("Cleared song feature cache for instant fresh indexing.")

if __name__ == '__main__':
    main()
