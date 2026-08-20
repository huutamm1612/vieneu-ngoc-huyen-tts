# VieNeu-TTS Ngọc Huyền

Pipeline training ba phase và inference tiếng Việt cho giọng đọc Ngọc Huyền. Phần lõi không phụ thuộc Modal: có thể clone repo để chạy bằng terminal, import trong Python/notebook, dùng Kaggle hoặc gọi GPU trên Modal.

**Model hoàn chỉnh sau Phase 3:** [huutamm1612/vieneu-tts-ngoc-huyen](https://huggingface.co/huutamm1612/vieneu-tts-ngoc-huyen)

- Base model: [pnnbao-ump/VieNeu-TTS](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
- Codec: [neuphonic/neucodec](https://huggingface.co/neuphonic/neucodec)
- Training data: [pnnbao-ump/ngochuyen_voice](https://huggingface.co/datasets/pnnbao-ump/ngochuyen_voice)
- Kaggle notebook: [notebooks/inference_kaggle.ipynb](notebooks/inference_kaggle.ipynb)

Model trên Hugging Face là **full model đã merge và fine-tune**, không phải LoRA adapter. Inference vẫn cần quyền truy cập gated model `neuphonic/neucodec`.

## Inference nhanh bằng terminal

### 1. Cài đặt

Yêu cầu Python `3.12`. GPU CUDA được khuyến nghị cho inference thực tế.

#### Windows PowerShell

```powershell
git clone https://github.com/huutamm1612/vieneu-ngoc-huyen-tts.git
cd vieneu-ngoc-huyen-tts

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[inference]"
```

Nếu PowerShell chặn script kích hoạt môi trường:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
git clone https://github.com/huutamm1612/vieneu-ngoc-huyen-tts.git
cd vieneu-ngoc-huyen-tts

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[inference]"
```

### 2. Đăng nhập Hugging Face

Tài khoản Hugging Face phải được cấp quyền sử dụng [NeuCodec](https://huggingface.co/neuphonic/neucodec). Sau đó đăng nhập bằng token có quyền đọc:

```powershell
hf auth login
```

Không ghi token trực tiếp vào source code hoặc commit token lên Git.

### 3. Chuẩn bị đầu vào

- Một file TXT chứa nội dung tiếng Việt cần đọc.
- Một reference WAV dài khoảng `3–15` giây, sạch và chỉ có một người nói.
- `reference-text` phải khớp chính xác với lời nói trong reference WAV.

### 4. Chạy inference

PowerShell:

```powershell
python -m inference `
  --config configs/inference.yaml `
  --model "huutamm1612/vieneu-tts-ngoc-huyen" `
  --input "D:\stories\story.txt" `
  --reference-audio "D:\voices\reference.wav" `
  --reference-text "Nội dung phải khớp chính xác với reference audio." `
  --output "D:\stories\story.wav"
```

Linux/macOS:

```bash
python -m inference \
  --config configs/inference.yaml \
  --model "huutamm1612/vieneu-tts-ngoc-huyen" \
  --input "/path/to/story.txt" \
  --reference-audio "/path/to/reference.wav" \
  --reference-text "Nội dung phải khớp chính xác với reference audio." \
  --output "/path/to/story.wav"
```

Model được tải trực tiếp từ Hugging Face và cache ở máy chạy. Pipeline tự chia văn bản dài, hiển thị tiến độ bằng `tqdm`, sinh từng segment rồi ghép đúng thứ tự thành **một file WAV duy nhất**.

Trên GPU có ít VRAM, giữ logical batch mặc định nhưng giới hạn mỗi forward pass:

```powershell
python -m inference ... --max-runtime-batch-size 8
```

Hai GPU:

```powershell
python -m inference ... --devices "cuda:0,cuda:1" --num-gpus 2
```

Mỗi GPU giữ một bản model và xử lý các batch độc lập; đây là data-parallel inference, không phải model sharding.

## Dùng trong Python hoặc notebook

Sau khi cài repo bằng `pip install -e ".[inference]"`:

```python
from inference import InferenceConfig, TTSInference

config = InferenceConfig(
    model="huutamm1612/vieneu-tts-ngoc-huyen",
    devices="auto",
    num_gpus=1,
    max_runtime_batch_size=16,
    show_progress=True,
)

with TTSInference(config) as tts:
    result = tts.infer(
        input_path="story.txt",
        reference_audio="reference.wav",
        reference_text="Nội dung phải khớp chính xác với reference audio.",
        output_path="story.wav",
    )

print(result.as_dict())
```

Nếu không truyền `batches`, `infer()` tự tiền xử lý với cấu hình mặc định:

```text
min_chars=80
target_chars=128
max_chars=156
batch_size=128
max_length_gap=12
```

Có thể gọi `prepare_batches(...)` trước nếu muốn kiểm tra hoặc chỉnh batch rồi mới truyền vào `tts.infer(batches=batches, ...)`.

## Chạy trên Kaggle

Notebook sẵn có: [notebooks/inference_kaggle.ipynb](notebooks/inference_kaggle.ipynb).

1. Upload notebook lên Kaggle và bật Internet + GPU.
2. Tạo Kaggle Secret `HF_TOKEN` có quyền đọc `neuphonic/neucodec`.
3. Nếu GitHub repo còn private, tạo thêm `GITHUB_TOKEN` có quyền đọc repo.
4. Chạy các cell, upload TXT bằng nút **Chọn TXT**.
5. Nghe hoặc tải WAV từ `/kaggle/working`.

Notebook clone code từ GitHub, tải model Phase 3 từ Hugging Face và sử dụng cùng API trong `src/inference`; notebook không chứa một bản inference riêng.

## Chạy Modal và tự tải WAV về máy

Modal chỉ là môi trường GPU. Code vẫn được lấy từ repo này.

```powershell
python -m pip install -e ".[cloud]"
Copy-Item .env.example .env.modal
notepad .env.modal
```

Điền `HF_TOKEN` vào `.env.modal`, sau đó đồng bộ sang Modal Secret:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\sync_modal_secrets.ps1 `
  -Environment "main"
```

Một lệnh sẽ upload TXT, chạy A10 trên Modal, đợi hoàn tất và tải WAV về đúng path local:

```powershell
python .\scripts\infer_modal_to_local.py `
  "D:\stories\story.txt" `
  "D:\stories\story.wav" `
  --model "huutamm1612/vieneu-tts-ngoc-huyen" `
  --reference-audio "/mnt/tts-dataset/dataset/audio/ngochuyen_00769.wav" `
  --reference-text "Nội dung phải khớp chính xác với reference audio." `
  --gpu "A10" `
  --num-gpus 1 `
  --env "main"
```

Reference audio trong ví dụ phải tồn tại trên Modal Volume. Bản WAV cuối được lưu ở path local thứ hai; một bản dự phòng vẫn được giữ trên Volume `tts-inference-results`.

## Pipeline inference

```text
TXT hoặc text
  -> chuẩn hóa tiếng Việt
  -> chia đoạn 80/128/156 ký tự
  -> sea-g2p
  -> gom batch theo độ dài
  -> encode reference một lần bằng NeuCodec
  -> sinh speech token
  -> NeuCodec decode
  -> sắp xếp theo index
  -> ghép thành một WAV
```

Nếu một segment vẫn lỗi sau số lần retry đã cấu hình, pipeline dừng và không tạo file WAV thiếu đoạn.

## Pipeline training ba phase

Một lần chạy trên Modal thực hiện tuần tự:

1. **Chuẩn bị dữ liệu:** T4 `16 GB`, 4 CPU, 6 GiB RAM mã hóa train/eval bằng NeuCodec và lưu cache lên Volume.
2. **Phase 1 — LoRA:** A10 `24 GB`, `lr=1e-4`, tối đa 5 epoch.
3. **Phase 2 — Safe merge:** nạp base BF16 sạch, gắn adapter, kiểm tra logits trước/sau merge và lưu full model.
4. **Phase 3 — Partial fine-tuning:** optimizer mới, mở 33% block trên cùng + output head/final norm, `lr=1e-6`, tối đa 2 epoch.

NeuCodec không tham gia optimizer. Phase 1 và Phase 3 đánh giá trên `metadata_eval.csv`, early stopping và nạp lại checkpoint có `eval_loss` tốt nhất trước khi xuất artifact cuối.

Chạy toàn bộ pipeline:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\train_modal.ps1 `
  -RunName "ngoc-a10-v1"
```

Override tham số mà không sửa code:

```powershell
.\scripts\train_modal.ps1 `
  -RunName "ngoc-a10-v2" `
  -Phase1LearningRate 0.0001 `
  -Phase1Epochs 5 `
  -Phase3LearningRate 0.000001 `
  -Phase3Epochs 2
```

Artifact cuối trên `tts-training-results`:

```text
/runs/<run-name>/
  resolved_config.yaml
  preparation_environment.json
  environment.json
  pipeline_state.json
  pipeline_summary.json
  phase1_lora/final/       # LoRA adapter tốt nhất
  phase2_merge/final/      # full model đã merge
  phase3_finetune/final/   # full model cuối cùng
```

Xem log và artifact:

```powershell
modal app logs --env main vieneu-tts-three-phase --follow
modal volume ls --env main tts-training-results /runs/ngoc-a10-v1
```

## Dataset contract

Dataset audio không nằm trong Git. Tải revision đã dùng để train và tái tạo cấu trúc local bằng:

```powershell
python -m pip install -e ".[data]"
python .\data\download_dataset.py
```

Mặc định script dùng revision bất biến `1ebbfbd1fb828dfd41bff8b1645ad9e56cfc614a` của
[`pnnbao-ump/ngochuyen_voice`](https://huggingface.co/datasets/pnnbao-ump/ngochuyen_voice),
đặt kết quả ngay trong `data/` và tạo đúng split `4.717 train + 96 eval`. Script hỗ trợ
resume, không xóa hoặc ghi đè sample đang có; file giống hệt được tái sử dụng, còn xung
đột sẽ làm chương trình dừng. Các shard nguồn vẫn nằm trong Hugging Face cache và mọi
mẫu bị loại khỏi tập train đều được ghi vào JSON reject.

Tạo dataset ở một vị trí khác:

```powershell
python .\data\download_dataset.py `
  --output-root "D:\datasets\ngochuyen_story_tts_clean"
```

Chỉ kiểm tra dataset đã có, không truy cập Hugging Face:

```powershell
python .\data\download_dataset.py --verify-only
```

Pipeline training đọc cấu trúc sau:

```text
/dataset/
  audio/*.wav
  metadata.csv          # filename.wav|text
  metadata_eval.csv     # filename.wav|text
  manifest.csv
  text_rejects.json
  audio_rejects.json
  processing_config.json
  processing_stats.json
  SOURCE.txt
```

Pipeline kiểm tra audio thiếu, filename trùng giữa train/eval và sequence dài quá 2.048 token. Audio được resample về 16 kHz trước khi NeuCodec encode.

## Cấu trúc repository

```text
configs/
  inference.yaml             # cấu hình inference
  pipeline_3phase.yaml       # hyperparameter training
cloud/
  modal_app.py               # NeuCodec preparation + ba phase training
  modal_inference.py         # inference trên Modal
  modal_hf_upload.py         # Modal Volume -> Hugging Face
scripts/
  infer.ps1 / infer.sh       # CLI inference local
  infer_modal_to_local.py    # TXT local -> Modal -> WAV local
  train_modal.ps1 / .sh      # khởi chạy training
  upload_model_to_hf.ps1     # upload Phase 3 lên Hugging Face
data/
  download_dataset.py        # Hugging Face -> dataset local sạch
  rename_dataset.py          # đổi prefix và đồng bộ metadata
src/inference/               # core inference độc lập môi trường
src/train/                   # core training độc lập Modal
notebooks/
  inference_kaggle.ipynb
tests/
```

## Đẩy Phase 3 lên Hugging Face

Uploader chạy bằng Modal CPU và truyền model trực tiếp từ Volume sang Hugging Face:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\upload_model_to_hf.ps1 `
  -RepoId "YOUR_HF_USERNAME/vieneu-tts-ngoc-huyen" `
  -RunName "ngoc-a10-v1" `
  -Environment "main"
```

Thêm `-Public` nếu chủ động muốn tạo model repository công khai. Token cần quyền ghi vào repository đích.

## Kiểm thử

Các kiểm thử này không tải model:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m compileall -q src cloud tests
```

## Giới hạn và sử dụng có trách nhiệm

- Model phù hợp nhất với tiếng Việt đã chuẩn hóa và văn phong đọc truyện tương tự dữ liệu train.
- Tên riêng, từ nước ngoài, chữ viết tắt và ký hiệu lạ có thể bị phát âm sai.
- Không sử dụng model để giả mạo người thật, lừa đảo hoặc tạo nội dung gây hiểu nhầm.
- Người dùng phải có quyền sử dụng giọng nói/reference audio và tuân thủ giấy phép của model, codec và dữ liệu.

Model được công bố với metadata `CC BY-NC 4.0` do ràng buộc của training dataset. Hãy xem model card trên [Hugging Face](https://huggingface.co/huutamm1612/vieneu-tts-ngoc-huyen) để biết chi tiết và attribution.
