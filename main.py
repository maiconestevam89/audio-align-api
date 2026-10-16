import io
import tempfile
import numpy as np
import librosa
import soundfile as sf
from scipy.spatial.distance import cdist
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="Audio Aligner API")

# Libera o acesso para o seu frontend (sem erro de CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_vocal_onset(y: np.ndarray, sr: int, hop_length: int = 512, frame_len: int = 1024) -> float:
    """
    Detecta o instante exato (em segundos) em que o vocal realmente comeca a cantar,
    ignorando ruidos de microfone, cliques, estalos ou respiracao inicial.
    """
    n_frames = (len(y) - frame_len) // hop_length + 1
    if n_frames <= 0:
        return 0.0
    
    rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = y[i * hop_length : i * hop_length + frame_len]
        rms[i] = np.sqrt(np.mean(seg**2))
        
    peak_rms = np.max(rms)
    if peak_rms < 0.008:
        return 0.0
        
    thresh = max(0.012, 0.06 * peak_rms)
    min_sustained_frames = max(2, int(0.070 * sr / hop_length))
    
    for i in range(len(rms) - min_sustained_frames):
        if np.all(rms[i : i + min_sustained_frames] > thresh):
            onset_frame = i
            while onset_frame > 0 and rms[onset_frame - 1] > 0.3 * thresh:
                onset_frame -= 1
            return max(0.0, onset_frame * hop_length / sr)
            
    return 0.0

def extract_features(y: np.ndarray, sr: int, hop_length: int = 512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    features = np.vstack([mfcc, mfcc_delta, chroma])
    norm = np.linalg.norm(features, axis=0, keepdims=True) + 1e-8
    return (features / norm).T

def wsola_stretch(audio: np.ndarray, warp_map: np.ndarray, sr: int, target_len: int, hop_length: int = 512) -> np.ndarray:
    n_samples = len(audio)
    win_len = int(0.040 * sr)
    hop_out = int(0.010 * sr)
    win = np.hanning(win_len).astype(np.float32)
    
    out = np.zeros(target_len + win_len, dtype=np.float32)
    norm = np.zeros(target_len + win_len, dtype=np.float32)
    
    t_out_axis = np.arange(0, target_len, hop_out)
    ref_sample_axis = np.arange(len(warp_map)) * hop_length
    interp_in_samples = np.interp(t_out_axis, ref_sample_axis, warp_map)
    
    for t_out, s_in in zip(t_out_axis, interp_in_samples):
        src_pos = int(np.clip(s_in, 0, n_samples - win_len))
        seg = audio[src_pos:src_pos + win_len]
        if len(seg) < win_len:
            continue
        out[t_out:t_out + win_len] += seg * win
        norm[t_out:t_out + win_len] += win
        
    mask = norm > 0.05
    out[mask] /= norm[mask]
    return out[:target_len]

def align_audio_core(ref_path: str, vocal_path: str, target_sr: int = 22050) -> bytes:
    y_ref, sr = librosa.load(ref_path, sr=target_sr, mono=True)
    y_voc, _ = librosa.load(vocal_path, sr=target_sr, mono=True)
    
    hop_length = 512
    
    # 1. Detecta o instante exato de entrada da voz da IA e da voz do cliente
    t_ref_onset = find_vocal_onset(y_ref, sr, hop_length)
    t_voc_onset = find_vocal_onset(y_voc, sr, hop_length)
    
    ref_start_idx = int(t_ref_onset * sr)
    voc_start_idx = int(t_voc_onset * sr)
    
    y_ref_active = y_ref[ref_start_idx:]
    y_voc_active = y_voc[voc_start_idx:]
    
    # Fallback seguro caso um dos trechos ativos seja excessivamente curto
    if len(y_ref_active) < 1.0 * sr or len(y_voc_active) < 1.0 * sr:
        aligned_audio = np.zeros(len(y_ref), dtype=np.float32)
        copy_len = min(len(y_voc_active), len(y_ref) - ref_start_idx)
        if copy_len > 0:
            aligned_audio[ref_start_idx : ref_start_idx + copy_len] = y_voc_active[:copy_len]
        buffer = io.BytesIO()
        sf.write(buffer, aligned_audio, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()
        
    # 2. Extrai caracteristicas a partir do ponto de entrada do canto
    feat_ref = extract_features(y_ref_active, sr, hop_length)
    feat_voc = extract_features(y_voc_active, sr, hop_length)
    
    dist_matrix = cdist(feat_ref, feat_voc, metric="cosine")
    
    # 3. DTW com penalidade suave para sincronizar frases sem distorcoes bruscas
    N, M = dist_matrix.shape
    D = np.full((N, M), np.inf, dtype=np.float32)
    D[0, 0] = dist_matrix[0, 0]
    
    for j in range(1, M):
        D[0, j] = D[0, j-1] + dist_matrix[0, j] * 1.5
    for i in range(1, N):
        D[i, 0] = D[i-1, 0] + dist_matrix[i, 0] * 1.5
        
    for i in range(1, N):
        for j in range(1, M):
            c = dist_matrix[i, j]
            d_diag = D[i-1, j-1] + c
            d_horiz = D[i, j-1] + c * 1.2
            d_vert = D[i-1, j] + c * 1.2
            D[i, j] = min(d_diag, d_horiz, d_vert)
            
    # Backtrack a partir do final
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            steps = [(i-1, j-1), (i, j-1), (i-1, j)]
            costs = [D[s] for s in steps]
            idx = np.argmin(costs)
            i, j = steps[idx]
        path.append((i, j))
        
    path = np.array(path[::-1])
    ref_frames, voc_frames = path[:, 0], path[:, 1]
    unique_ref, idx = np.unique(ref_frames, return_index=True)
    aligned_voc_frames = voc_frames[idx]
    
    all_ref_frames = np.arange(N)
    full_voc_frames = np.interp(all_ref_frames, unique_ref, aligned_voc_frames)
    
    # Time-stretching do canto ativo
    warp_sample_map = full_voc_frames * hop_length
    stretched_active = wsola_stretch(y_voc_active, warp_sample_map, sr, target_len=len(y_ref_active), hop_length=hop_length)
    
    # 4. Monta a faixa completa: silencio durante a intro instrumental,
    # e a voz do cliente entra exatamente no mesmo milissegundo da voz da IA
    aligned_audio = np.zeros(len(y_ref), dtype=np.float32)
    fit_len = min(len(stretched_active), len(y_ref) - ref_start_idx)
    if fit_len > 0:
        aligned_audio[ref_start_idx : ref_start_idx + fit_len] = stretched_active[:fit_len]
        
    buffer = io.BytesIO()
    sf.write(buffer, aligned_audio, sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()

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
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_ref,              tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_voc:
            f_ref.write(ref_bytes)
            f_voc.write(voc_bytes)
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
