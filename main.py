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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(y: np.ndarray, sr: int, hop_length: int = 512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    features = np.vstack([mfcc, mfcc_delta, chroma])
    norm = np.linalg.norm(features, axis=0, keepdims=True) + 1e-8
    return (features / norm).T

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
    # Carrega mantendo a taxa de amostragem idêntica
    y_ref, sr = librosa.load(ref_path, sr=target_sr, mono=True)
    y_voc, _ = librosa.load(vocal_path, sr=target_sr, mono=True)
    
    # 1. Detectar o ataque inicial (Início do canto na IA)
    # Procura onde o áudio da IA realmente começa a ter energia (ignora o silêncio inicial)
    non_silent_ref = librosa.effects.split(y_ref, top_db=35)
    start_sample_ref = non_silent_ref[0][0] if len(non_silent_ref) > 0 else 0
    
    non_silent_voc = librosa.effects.split(y_voc, top_db=35)
    start_sample_voc = non_silent_voc[0][0] if len(non_silent_voc) > 0 else 0

    # Trima o silêncio inicial de ambos para alinhar apenas o conteúdo cantado
    y_ref_trimmed = y_ref[start_sample_ref:]
    y_voc_trimmed = y_voc[start_sample_voc:]

    # 2. Extração de Features no trecho ativo
    hop_length = 512
    feat_ref = extract_features(y_ref_trimmed, sr, hop_length)
    feat_voc = extract_features(y_voc_trimmed, sr, hop_length)
    
    # 3. DTW no trecho ativo
    dist_matrix = cdist(feat_ref, feat_voc, metric="cosine")
    _, wp = librosa.sequence.dtw(C=dist_matrix, subseq=False, backtrack=True)
    wp = wp[::-1]
    
    ref_frames, voc_frames = wp[:, 0], wp[:, 1]
    unique_ref, idx = np.unique(ref_frames, return_index=True)
    aligned_voc_frames = voc_frames[idx]
    
    n_ref_frames = len(feat_ref)
    all_ref_frames = np.arange(n_ref_frames)
    full_voc_frames = np.interp(all_ref_frames, unique_ref, aligned_voc_frames)
    
    warp_sample_map = full_voc_frames * hop_length
    aligned_active = wsola_stretch(y_voc_trimmed, warp_sample_map, sr, target_len=len(y_ref_trimmed))
    
    # 4. Reconstrução com o silêncio exato da IA e duração total idêntica
    final_audio = np.zeros(len(y_ref), dtype=np.float32)
    
    # Insere o áudio processado exatamente a partir do segundo que a IA começa a cantar
    end_sample = min(start_sample_ref + len(aligned_active), len(y_ref))
    insert_length = end_sample - start_sample_ref
    final_audio[start_sample_ref:end_sample] = aligned_active[:insert_length]
    
    # Exporta em WAV com a exata mesma taxa e tempo da IA
    buffer = io.BytesIO()
    sf.write(buffer, final_audio, sr, format="WAV", subtype="PCM_16")
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
