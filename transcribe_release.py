import os, sys, json, urllib.request, urllib.parse, base64, re, subprocess

token = os.environ['GH_TOKEN']
repo = os.environ['GH_REPO']

gh = {'Accept':'application/vnd.github+json','Authorization':'Bearer '+token}

# Step 1: Scan ALL releases for untranscribed .wav files
print('Scanning all releases for untranscribed audio...')
release_jobs = []
page = 1
while True:
    url = 'https://api.github.com/repos/%s/releases?per_page=100&page=%d' % (repo, page)
    rels = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=gh)).read())
    if not rels:
        break
    for rel in rels:
        upload_url = rel.get('upload_url', '')
        existing_names = {a['name'] for a in rel.get('assets', [])}
        for a in rel.get('assets', []):
            name = a['name']
            if not name.endswith('.wav'):
                continue
            base = name.rsplit('.', 1)[0]
            if base + '.srt' not in existing_names:
                                release_jobs.append((a, upload_url, existing_names))
    page += 1

if not release_jobs:
    print('No new audio files to transcribe')
    exit(0)

print('Found %d audio file(s) to transcribe' % len(release_jobs))

# Step 3: Download model (once)
from funasr import AutoModel
print('Loading SenseVoiceSmall model...')
model = AutoModel(model='iic/SenseVoiceSmall', vad_model='fsmn-vad', punc_model='ct-punc',
                  spk_model=None, disable_update=True, device='cpu')
print('Model loaded')

# Step 4: Transcribe each
for asset, upload_url_template, existing_names in release_jobs:
    try:
        name = asset['name']
        base = name.rsplit('.', 1)[0]
        download_url = asset['browser_download_url']
        wav_path = '/tmp/' + name
        print('Downloading: %s (%d MB)' % (name, asset['size'] // 1024 // 1024))
        urllib.request.urlretrieve(download_url, wav_path)

        # Get audio duration via ffprobe
        dur_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', wav_path]
        dur_result = subprocess.run(dur_cmd, capture_output=True, text=True)
        total_sec = float(dur_result.stdout.strip() or 0)
        total_sec = int(total_sec)
        print('Audio duration: %ds (%dm%ds)' % (total_sec, total_sec // 60, total_sec % 60))

        # Split into 10-minute segments (600s)
        segment_sec = 600
        text_lines = []
        srt_lines = []
        srt_idx = 1
        seg_offset = 0
        seg_num = 0

        while seg_offset < total_sec:
            seg_end = min(seg_offset + segment_sec, total_sec)
            seg_path = '/tmp/seg_%d.wav' % seg_num
            seg_cmd = ['ffmpeg', '-y', '-loglevel', 'warning', '-i', wav_path,
                       '-ss', str(seg_offset), '-to', str(seg_end),
                       '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', seg_path]
            subprocess.run(seg_cmd)
            seg_len = seg_end - seg_offset
            print('  Segment %d: %dm%ds - %dm%ds (%ds)' % (seg_num + 1, seg_offset // 60, seg_offset % 60, seg_end // 60, seg_end % 60, seg_len))

            result = model.generate(input=seg_path, ban_emo_unk=True, cache={})
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        txt = item.get('text', '') or item.get('sentence', '') or ''
                        if txt.strip():
                            txt = re.sub(r'<\|[^|]+\|>\s*', '', txt).strip()
                            if txt:
                                text_lines.append(txt)
                            ts = item.get('timestamp', '')
                            if ts:
                                if isinstance(ts, list) and len(ts) > 0:
                                    for seg in ts:
                                        if isinstance(seg, list) and len(seg) >= 3:
                                            st_ms, et_ms, seg_txt_raw = int(seg[0]), int(seg[1]), seg[2]
                                            seg_txt = re.sub(r'<\|[^|]+\|>\s*', '', seg_txt_raw).strip()
                                            if not seg_txt:
                                                continue
                                            st_ms += seg_offset * 1000
                                            et_ms += seg_offset * 1000
                                            st_s = st_ms // 1000
                                            st_fmt = '%02d:%02d:%02d,%03d' % (st_s // 3600, (st_s % 3600) // 60, st_s % 60, st_ms % 1000)
                                            et_s = et_ms // 1000
                                            et_fmt = '%02d:%02d:%02d,%03d' % (et_s // 3600, (et_s % 3600) // 60, et_s % 60, et_ms % 1000)
                                            srt_lines.append('%d\n%s --> %s\n%s\n' % (srt_idx, st_fmt, et_fmt, seg_txt))
                                            srt_idx += 1
                    elif isinstance(item, str) and item.strip():
                        item = re.sub(r'<\|[^|]+\|>\s*', '', item).strip()
                        if item:
                            text_lines.append(item)
            elif isinstance(result, dict):
                txt = result.get('text', '') or result.get('sentence', '') or ''
                if txt.strip():
                    txt = re.sub(r'<\|[^|]+\|>\s*', '', txt).strip()
                    if txt:
                        text_lines.append(txt)

            os.remove(seg_path)
            seg_offset = seg_end
            seg_num += 1

        os.remove(wav_path)
        if not text_lines:
            print('  No transcription text for %s' % name)
            continue

        # Upload TXT (skip if already exists)
        txt_name = base + '.txt'
        if txt_name not in existing_names:
            txt_text = '\n'.join(text_lines)
            upload_url = upload_url_template.replace('{?name,label}', '?name=' + urllib.parse.quote(txt_name))
            req = urllib.request.Request(upload_url,
                data=txt_text.encode('utf-8'),
                headers=dict(gh, **{'Content-Type': 'text/plain; charset=utf-8'}),
                method='POST')
            urllib.request.urlopen(req, timeout=120)
            print('  Uploaded: %s' % txt_name)
        else:
            print('  Skipping %s (already exists)' % txt_name)

        # Upload SRT if we have timestamps
        if srt_lines:
            srt_name = base + '.srt'
            if srt_name not in existing_names:
                srt_text = ''.join(srt_lines)
                upload_url2 = upload_url_template.replace('{?name,label}', '?name=' + urllib.parse.quote(srt_name))
                req2 = urllib.request.Request(upload_url2,
                    data=srt_text.encode('utf-8'),
                    headers=dict(gh, **{'Content-Type': 'text/plain; charset=utf-8'}),
                    method='POST')
                urllib.request.urlopen(req2, timeout=120)
                print('  Uploaded: %s' % srt_name)
            else:
                print('  Skipping %s (already exists)' % srt_name)

        print('Transcription complete')
    except Exception as e:
        print('  Error transcribing %s: %s' % (name, e))
        continue
