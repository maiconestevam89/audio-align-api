import io
import tempfile
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="Audio Aligner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def align_audio_core(ref_path: str, vocal_path: str, target_sr: int = 22050) -> bytes:
    """
    Alinha o audio do cliente para ter o MESMO tempo total e sincronismo metrico
    do audio de referencia da IA (Suno), respeitando estritamente o silencio inicial
    e pausas instrumentais.
    """
    y_ref, sr = librosa.load(ref_path, sr=target_sr, mono=True)
    y_voc, _ = librosa.load(vocal_path, sr=target_sr, mono=True)
    
    hop_length = int(0.040 * sr)  # 40ms hop para sincronismo preciso e rapido
    win_length = int(0.080 * sr)  # 80ms analise
    
    # 1. Detecta o instante exato em que a voz da IA comeca a cantar no arranjo.
    # Usa RMS sustentado com threshold relativo robusto contra bleed de bateria/instrumentos.
    rms_ref = librosa.feature.rms(y=y_ref, hop_length=hop_length, frame_length=win_length)[0]
    peak_ref_rms = float(np.max(rms_ref)) if len(rms_ref) > 0 else 0.0
    ref_thresh = max(0.020, 0.15 * peak_ref_rms)
    min_sustained = max(2, int(0.20 * sr / hop_length))  # minimo 200ms sustentado
    
    ref_onset_frame = 0
    for i in range(len(rms_ref) - min_sustained):
        if np.all(rms_ref[i : i + min_sustained] > ref_thresh):
            ref_onset_frame = i
            # Recua suavemente para o inicio do ataque/respiracao
            while ref_onset_frame > 0 and rms_ref[ref_onset_frame - 1] > 0.4 * ref_thresh:
                ref_onset_frame -= 1
            break
            
    t_ref_onset = ref_onset_frame * hop_length / sr
    ref_start_idx = int(t_ref_onset * sr)
    
    # 2. Detecta o inicio e o fim do canto na gravacao do cliente (remove silencio inicial da gravacao)
    rms_voc = librosa.feature.rms(y=y_voc, hop_length=hop_length, frame_length=win_length)[0]
    peak_voc_rms = float(np.max(rms_voc)) if len(rms_voc) > 0 else 0.0
    voc_thresh = max(0.015, 0.12 * peak_voc_rms)
    
    voc_onset_frame = 0
    for i in range(len(rms_voc) - min_sustained):
        if np.all(rms_voc[i : i + min_sustained] > voc_thresh):
            voc_onset_frame = i
            while voc_onset_frame > 0 and rms_voc[voc_onset_frame - 1] > 0.4 * voc_thresh:
                voc_onset_frame -= 1
            break
            
    voc_end_frame = len(rms_voc)
    for i in range(len(rms_voc) - 1, voc_onset_frame, -1):
        if rms_voc[i] > voc_thresh:
            voc_end_frame = min(len(rms_voc), i + int(0.50 * sr / hop_length))
            break
            
    voc_start_idx = int(voc_onset_frame * hop_length)
    voc_end_idx = min(len(y_voc), int(voc_end_frame * hop_length))
    
    y_voc_active = y_voc[voc_start_idx : voc_end_idx]
    y_ref_active = y_ref[ref_start_idx :]
    
    # Se algum trecho for nulo ou excessivamente curto, posiciona no ponto exato da entrada
    if len(y_voc_active) < int(0.5 * sr) or len(y_ref_active) < int(0.5 * sr):
        aligned_audio = np.zeros(len(y_ref), dtype=np.float32)
        fit_len = min(len(y_voc), len(y_ref) - ref_start_idx)
        if fit_len > 0:
            aligned_audio[ref_start_idx : ref_start_idx + fit_len] = y_voc[:fit_len]
        buf = io.BytesIO()
        sf.write(buf, aligned_audio, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
        
    # 3. Extracao de caracteristicas (MFCCs para fonemas + Chroma STFT para melodia)
    mfcc_ref = librosa.feature.mfcc(y=y_ref_active, sr=sr, n_mfcc=13, hop_length=hop_length)
    chroma_ref = librosa.feature.chroma_stft(y=y_ref_active, sr=sr, hop_length=hop_length)
    feat_ref = np.vstack([mfcc_ref, chroma_ref])
    
    mfcc_voc = librosa.feature.mfcc(y=y_voc_active, sr=sr, n_mfcc=13, hop_length=hop_length)
    chroma_voc = librosa.feature.chroma_stft(y=y_voc_active, sr=sr, hop_length=hop_length)
    feat_voc = np.vstack([mfcc_voc, chroma_voc])
    
    # DTW compilado e ultra-rapido (librosa numba)
    try:
        D, wp = librosa.sequence.dtw(X=feat_voc, Y=feat_ref, metric="cosine", backtrack=True)
        wp = wp[::-1]  # ordem cronologica
        voc_frames = wp[:, 0]
        ref_frames = wp[:, 1]
        
        unique_ref, idx = np.unique(ref_frames, return_index=True)
        aligned_voc_frames = voc_frames[idx]
        
        all_ref_frames = np.arange(feat_ref.shape[1])
        full_voc_frames = np.interp(all_ref_frames, unique_ref, aligned_voc_frames)
    except Exception:
        # Fallback linear suave caso DTW falhe
        all_ref_frames = np.arange(feat_ref.shape[1])
        full_voc_frames = np.linspace(0, feat_voc.shape[1] - 1, len(all_ref_frames))
        
    # 4. Time-Stretching WSOLA para sincronizar metricamente sem alterar o tom (pitch)
    Hs = int(0.020 * sr)  # salto de sintese de 20ms
    N = Hs * 2           # janela de 40ms
    tol = int(0.012 * sr) # faixa de correlacao de forma de onda
    D_dec = 4
    
    n_voc_samples = len(y_voc_active)
    target_active_len = len(y_ref_active)
    
    nd = n_voc_samples // D_dec
    L = Hs // D_dec
    mono_voc = y_voc_active[::D_dec][:nd].astype(np.float32)
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)).astype(np.float32)
    
    stretched_active = np.zeros(target_active_len + 3 * N, dtype=np.float32)
    norm = np.zeros(target_active_len + 3 * N, dtype=np.float32)
    
    warp_sample_map = full_voc_frames * hop_length
    ref_sample_axis = all_ref_frames * hop_length
    last_frame = int(np.ceil(target_active_len / Hs))
    prev_start = 0
    have_prev = False
    
    for k in range(-1, last_frame + 1):
        p = k * Hs
        s_in = float(np.interp(p + Hs, ref_sample_axis, warp_sample_map))
        if s_in < 0 or s_in >= n_voc_samples:
            have_prev = False
            continue
            
        target = int(round(s_in)) - Hs
        start = target
        
        if have_prev:
            tpl = int(round((prev_start + Hs) / D_dec))
            if 0 <= tpl and tpl + L <= nd:
                tpl_seg = mono_voc[tpl : tpl + L]
                tpl_e = float(np.sum(tpl_seg**2)) + 1e-9
                lo = max(0, int(round((target - tol) / D_dec)))
                hi = min(nd - L, int(round((target + tol) / D_dec)))
                best_score = -1e9
                best_sd = -1
                for sd in range(lo, hi + 1):
                    cand = mono_voc[sd : sd + L]
                    dot = float(np.sum(tpl_seg * cand))
                    e = float(np.sum(cand**2)) + 1e-9
                    score = dot / np.sqrt(e * tpl_e) - (0.02 * abs(sd * D_dec - target)) / tol
                    if score > best_score:
                        best_score = score
                        best_sd = sd
                if best_sd >= 0:
                    start = best_sd * D_dec
                    
        for m in range(N):
            o = p + m
            idx_s = start + m
            if 0 <= o < len(stretched_active) and 0 <= idx_s < n_voc_samples:
                stretched_active[o] += win[m] * y_voc_active[idx_s]
                norm[o] += win[m]
                
        prev_start = start
        have_prev = True
        
    mask = norm > 0.05
    stretched_active[mask] /= norm[mask]
    stretched_active = stretched_active[:target_active_len]
    
    # 5. Montagem da faixa final com a EXATA mesma duracao da referencia
    # [0 : ref_start_idx] e puro silencio (zeros)
    # [ref_start_idx : ] e a voz do cliente cantando no tempo exato
    aligned_audio = np.zeros(len(y_ref), dtype=np.float32)
    fit_len = min(len(stretched_active), len(y_ref) - ref_start_idx)
    if fit_len > 0:
        aligned_audio[ref_start_idx : ref_start_idx + fit_len] = stretched_active[:fit_len]
        
    buf = io.BytesIO()
    sf.write(buf, aligned_audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Audio Aligner API"}

@app.post("/align")
def align_endpoint(
    audio_referencia: UploadFile = File(...),
    audio_cliente: UploadFile = File(...)
):
    try:
        ref_bytes = audio_referencia.file.read()
        voc_bytes = audio_cliente.file.read()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_ref, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_voc:
            f_ref.write(ref_bytes)
            f_voc.write(voc_bytes)
            ref_path = f_ref.name
            vocal_path = f_voc.name
        
        aligned_wav_bytes = align_audio_core(ref_path, vocal_path)
        
        return StreamingResponse(
            io.BytesIO(aligned_wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=vocal_alinhado.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
