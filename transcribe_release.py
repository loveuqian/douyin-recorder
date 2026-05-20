import os, sys, json, urllib.request, urllib.parse, base64, re, subprocess

token = os.environ['GH_TOKEN']
repo = os.environ['GH_REPO']

gh = {'Accept':'application/vnd.github+json','Authorization':'Bearer '+token}

# Step 1: Get filename from env (set by workflow dispatch) or scan untranscribed
input_filename = os.environ.get('TRANSCRIBE_FILENAME', '')
release_jobs = []
if input_filename:
    print('Targeting file:', input_filename)
    # Find this specific file in releases
    page = 1
    while True:
        url = 'https://api.github.com/repos/%s/releases?per_page=100&page=%d' % (repo, page)
        rels = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=gh)).read())
        if not rels:
            break
        for rel in rels:
            upload_url = rel.get('upload_url', '')
            for a in rel.get('assets', []):
                if a['name'] == input_filename:
                    existing_names = {aa['name'] for aa in rel.get('assets', [])}
                    release_jobs.append((a, upload_url, existing_names))
                    break
        page += 1
else:
    # No filename given (e.g. manual trigger), skip - old files not processed
    print('No TRANSCRIBE_FILENAME set, nothing to do')
    exit(0)

if not release_jobs:
    print('Target file not found:', input_filename)
    exit(0)

print('Found %d file(s) to transcribe' % len(release_jobs))

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

            result = model.generate(input=seg_path, ban_emo_unk=True, cache={}, vad_kwargs={"max_single_segment_duration": 5000, "max_end_silence": 400})
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        txt = item.get('text', '') or item.get('sentence', '') or ''
                        if txt.strip():
                            txt = re.sub(r'<\s*\|[^|]+\|\s*>\s*', '', txt).strip()
                            if txt:
                                text_lines.append(txt)
                            ts = item.get('timestamp', '')
                            if ts:
                                if isinstance(ts, list) and len(ts) > 0:
                                    for seg in ts:
                                        if isinstance(seg, list) and len(seg) >= 3:
                                            st_ms, et_ms, seg_txt_raw = int(seg[0]), int(seg[1]), seg[2]
                                            seg_txt = re.sub(r'<\s*\|[^|]+\|\s*>\s*', '', seg_txt_raw).strip()
                                            if not seg_txt:
                                                continue
                                            seg_dur = (et_ms - st_ms) / 1000.0
                                            seg_chars = len(seg_txt)
                                            # Sub-split long segments into ~3s chunks
                                            if seg_dur > 6.0 and seg_chars > 0:
                                                punct = re.split('[。！？]', seg_txt)
                                                punct = [p.strip() for p in punct if p.strip()]
                                                if len(punct) >= 2:
                                                    cum_chars = 0
                                                    for pi, part in enumerate(punct):
                                                        plen = len(part)
                                                        ps = int(st_ms + (cum_chars / seg_chars) * (et_ms - st_ms))
                                                        pe = int(st_ms + ((cum_chars + plen) / seg_chars) * (et_ms - st_ms))
                                                        if pe <= ps: pe = ps + 500
                                                        st_fmt = '%02d:%02d:%02d,%03d' % (ps // 3600000, (ps // 60000) % 60, (ps // 1000) % 60, ps % 1000)
                                                        et_fmt = '%02d:%02d:%02d,%03d' % (pe // 3600000, (pe // 60000) % 60, (pe // 1000) % 60, pe % 1000)
                                                        srt_lines.append('%d\n%s --> %s\n%s\n' % (srt_idx, st_fmt, et_fmt, part))
                                                        srt_idx += 1
                                                        cum_chars += plen
                                                else:
                                                    sub_chars = max(1, int(seg_chars * 3.0 / seg_dur))
                                                    for kk in range(0, seg_chars, sub_chars):
                                                        sub_txt = seg_txt[kk:kk+sub_chars]
                                                        frac_s = kk / seg_chars
                                                        frac_e = (kk + len(sub_txt)) / seg_chars
                                                        sub_st = int(st_ms + frac_s * (et_ms - st_ms))
                                                        sub_et = int(st_ms + frac_e * (et_ms - st_ms))
                                                        if sub_et <= sub_st: sub_et = sub_st + 500
                                                        st_fmt = '%02d:%02d:%02d,%03d' % (sub_st // 3600000, (sub_st // 60000) % 60, (sub_st // 1000) % 60, sub_st % 1000)
                                                        et_fmt = '%02d:%02d:%02d,%03d' % (sub_et // 3600000, (sub_et // 60000) % 60, (sub_et // 1000) % 60, sub_et % 1000)
                                                        srt_lines.append('%d\n%s --> %s\n%s\n' % (srt_idx, st_fmt, et_fmt, sub_txt))
                                                        srt_idx += 1

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
