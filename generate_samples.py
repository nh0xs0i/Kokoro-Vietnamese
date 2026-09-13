"""
Tạo trước các file mẫu giọng đọc cho Kokoro-Vietnamese
Lưu vào thư mục sample_voices/
"""
from pathlib import Path
import time
import soundfile as sf
import torch

from kokoro_vietnamese import VOICES
from kokoro_vietnamese._kokoro import KModel
from kokoro_vietnamese.core import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_MODEL_FILE,
    SAMPLE_RATE,
    phonemize,
    normalize_audio,
    prepare_transformers_for_kokoro,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "sample_voices"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_TEXTS = {
    # Mẫu ngắn gọn, tự nhiên cho từng giọng
    "default": "Xin chào, đây là giọng đọc thử nghiệm của {label}."
}

def main():
    print("=" * 60)
    print("Đang chuẩn bị mô hình Kokoro...")
    prepare_transformers_for_kokoro()

    model_path = BASE_DIR / DEFAULT_MODEL_FILE
    config_path = BASE_DIR / DEFAULT_CONFIG_FILE

    t0 = time.time()
    model = KModel(
        repo_id='hexgrad/Kokoro-82M',
        config=str(config_path),
        model=str(model_path),
    ).to("cpu").eval()
    print(f"Mô hình đã tải xong trong {time.time() - t0:.2f}s")

    for voice_code, info in VOICES.items():
        out_file = OUTPUT_DIR / f"{voice_code}.wav"
        if out_file.exists():
            print(f"[-] Đã có mẫu cho {voice_code} ({info['label']}), bỏ qua.")
            continue

        label = info.get("label", voice_code)
        text = f"Xin chào, đây là giọng đọc mẫu của {label}."
        voicepack_path = BASE_DIR / "voicepacks" / f"{voice_code}.pt"

        if not voicepack_path.exists():
            print(f"[!] Không tìm thấy file {voicepack_path}, bỏ qua.")
            continue

        print(f"[+] Đang tạo mẫu cho {voice_code} ({label})...")
        t_gen = time.time()
        voicepack = torch.load(voicepack_path, map_location='cpu', weights_only=True)

        ps = phonemize(text)
        with torch.no_grad():
            ref_s = voicepack[len(ps) - 1]
            audio = model(ps, ref_s, 1.0)

        audio_np = audio.detach().cpu().numpy()
        audio_np = normalize_audio(audio_np, 0.95)

        sf.write(str(out_file), audio_np, SAMPLE_RATE)
        print(f"    ✓ Xong trong {time.time() - t_gen:.2f}s -> {out_file.name}")

    print("=" * 60)
    print("Hoàn tất tạo tất cả file mẫu giọng đọc!")

if __name__ == "__main__":
    main()
