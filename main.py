import io
import tempfile
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="Audio Aligner API - SyncVocal Architecture")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INF = 1e30
STEP_PENALTY = 0.15

def extract_syncvocal_features(y: np.ndarray, sr: int = 22050, hop_length: int = 512):
    """
    Extrai MFCCs, Chroma, Pitch (YIN) e Máscara de Atividade Vocal (VAD)
    """
    n_frames = int(np.ceil(len(y) / hop_length))
    
    # 1. MFCCs (13 coeficientes)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    # 2. Chroma CENS (12 bandas harmônicas)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    
    # 3. Estimador de Pitch Rápido (YIN)
    fmin = float(librosa.note_to_hz('C2'))
    fmax = float(librosa.note_to_hz('C7'))
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    
    f0_clean = np.nan_to_num(f0, nan=0.0)
    voiced = f0_clean > 0
    norm_f0 = np.zeros(len(f0_clean), dtype=np.float32)
    if np.any(voiced):
        log_f0 = np.log2(f0_clean[voiced])
        std = np.std(log_f0)
        norm_f0[voiced] = (log_f0 - np.mean(log_f0)) / (std if std > 1e-5 else 1.0)
    norm_f0 = norm_f0.reshape(1, -1)
    
    # Combina as características
    raw_feats = np.vstack([mfcc, chroma, norm_f0])
    
    # Ajuste de tamanho de quadros (frames)
    if raw_feats.shape[1] > n_frames:
        raw_feats = raw_feats[:, :n_frames]
    elif raw_feats.shape[1] < n_frames:
        pad_width = n_frames - raw_feats.shape[1]
        raw_feats = np.pad(raw_feats, ((0, 0), (0, pad_width)))
        
    # Normalização L2 por frame
    norm = np.linalg.norm(raw_feats, axis=0, keepdims=True) + 1e-8
    feats = (raw_feats / norm).T
    
    # 4. Detecção de Voz Ativa (Active Mask / VAD)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    thresh = np.max(rms) * 0.04
    active = rms > thresh
    
    if len(active) < n_frames:
        active = np.pad(active, (0, n_frames - len(active)))
    else:
        active = active[:n_frames]
        
    return feats, active

def subsequence_dtw_syncvocal(q_feats, q_active, r_feats, r_active):
    """
    Subsequence DTW ultra-otimizado (vetorizado em NumPy)
    q: Voz do Cliente (Query)
    r: Áudio da IA (Reference - Início e Fim Livres)
    """
    N, dim = q_feats.shape
    M, _ = r_feats.shape
    bp = np.zeros((N, M), dtype=np.uint8)

    # Pré-calcula matriz de similaridade de cosseno
    dots_matrix = np.dot(q_feats, r_feats.T) # (N, M)

    Dm2 = np.full(M, INF, dtype=np.float32)
    Dm1 = np.full(M, INF, dtype=np.float32)
    
    # Custo inicial do primeiro frame do Query
    if not q_active[0]:
        cPrev = np.where(~r_active, 0.4, 1.0).astype(np.float32)
    else:
        cPrev = np.where(~r_active, 1.0, 1.0 - dots_matrix[0]).astype(np.float32)
        
    Dm1[:] = cPrev
    bp[0, :] = 255

    for i in range(1, N):
        if not q_active[i]:
            cCur = np.where(~r_active, 0.4, 1.0).astype(np.float32)
        else:
            cCur = np.where(~r_active, 1.0, 1.0 - dots_matrix[i]).astype(np.float32)
            
        # Vetorização das 3 transições do Subsequence DTW
        val0 = np.full(M, INF, dtype=np.float32)
        val1 = np.full(M, INF, dtype=np.float32)
        val2 = np.full(M, INF, dtype=np.float32)

        # Passada 0: (i-1, j-1) -> (i, j)
        val0[1:] = Dm1[:-1] + cCur[1:]

        # Passada 1: (i-1, j-2) -> (i, j)
        val1[2:] = Dm1[:-2] + 0.5 * (cCur[1:-1] + cCur[2:]) + STEP_PENALTY

        # Passada 2: (i-2, j-1) -> (i, j)
        if i >= 2:
            val2[1:] = Dm2[:-1] + cPrev[1:] + cCur[1:] + STEP_PENALTY

        # Seleção do menor custo
        best = val0.copy()
        step = np.zeros(M, dtype=np.uint8) # Default step 0

        mask1 = val1 < best
        best[mask1] = val1[mask1]
        step[mask1] = 1

        mask2 = val2 < best
        best[mask2] = val2[mask2]
        step[mask2] = 2

        invalid = best >= INF
        step[invalid] = 3

        bp[i, :] = step
        Dm2[:] = Dm1
        Dm1[:] = best
        cPrev = cCur

    bestJ = np.argmin(Dm1)
    if Dm1[bestJ] >= INF:
        return None

    # Reconstrução do caminho (Backtracking)
    rev = []
    i, j = N - 1, int(bestJ)
    while True:
        rev.append((i, j))
        s = bp[i, j]
        if s == 255 or s == 3:
            break
        if s == 0:
            i -= 1
            j -= 1
        elif s == 1:
            rev.append((i, j - 1))
            i -= 1
            j -= 2
        elif s == 2:
            rev.append((i - 1, j))
            i -= 2
            j -= 1
        else:
            break

    rev.reverse()
    return np.array(rev)

def wsola_stretch(audio: np.ndarray, warp_map: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    n_samples = len(audio)
    win_len = int(0.040 * sr)
    hop_out = int(0.020 * sr)
    win = np.hanning(win_len)
    
    out = np.zeros(target_len + win_len, dtype=np.float32)
    norm = np.zeros(target_len + win_len, dtype=np.float32)
    
    t_out_axis = np.arange(0, target_len, hop_out)
    interp_in_samples = np.interp(t_out_axis, np.arange(len(warp_map)), warp_map)
    
    for t_out, s_in in zip(t_out_axis, interp_in_samples):
        src_pos = int(np.clip(s_in, 0, n_samples - win_len))
        seg = audio[src_pos:src_pos + win_len]
        if len(seg) < win_len:
            continue
        out[t_out:t_out + win_len] += seg * win
        norm[t_out:t_out + win_len] += win ** 2
        
    mask = norm > 1e-4
    out[mask] /= norm[mask]
    return out[:target_len]

def align_audio_core(ref_path: str, vocal_path: str, target_sr: int = 22050) -> bytes:
    y_ref, sr = librosa.load(ref_path, sr=target_sr, mono=True)
    y_voc, _ = librosa.load(vocal_path, sr=target_sr, mono=True)
    
    hop_length = 512
    q_feats, q_active = extract_syncvocal_features(y_voc, sr, hop_length) # Cliente
    r_feats, r_active = extract_syncvocal_features(y_ref, sr, hop_length) # IA
    
    path = subsequence_dtw_syncvocal(q_feats, q_active, r_feats, r_active)
    if path is None:
        raise ValueError("Falha ao alinhar as faixas de áudio.")
        
    q_frames, r_frames = path[:, 0], path[:, 1]
    
    unique_r, idx = np.unique(r_frames, return_index=True)
    matched_q = q_frames[idx]
    
    all_r_frames = np.arange(len(r_feats))
    full_q_frames = np.interp(all_r_frames, unique_r, matched_q)
    
    warp_sample_map = full_q_frames * hop_length
    aligned_audio = wsola_stretch(y_voc, warp_sample_map, sr, target_len=len(y_ref))
    
    # Silencia o trecho antes da entrada do canto (introdução instrumental)
    start_sample = int(unique_r[0] * hop_length)
    if start_sample > 0:
        aligned_audio[:start_sample] = 0.0
        
    # Silencia após o fim do canto (outro instrumental)
    end_sample = int(unique_r[-1] * hop_length)
    if end_sample < len(aligned_audio):
        aligned_audio[end_sample:] = 0.0
        
    buffer = io.BytesIO()
    sf.write(buffer, aligned_audio, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Audio Aligner API"}

@app.post("/align")
async def align_endpoint(
    audio_referencia: UploadFile = File(...),
    audio_cliente: UploadFile = File(...)
):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_ref, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_voc:
            f_ref.write(await audio_referencia.read())
            f_voc.write(await audio_cliente.read())
            ref_path = f_ref.name
            voc_path = f_voc.name
        
        aligned_wav_bytes = align_audio_core(ref_path, voc_path)
        
        return StreamingResponse(
            io.BytesIO(aligned_wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=vocal_alinhado.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
