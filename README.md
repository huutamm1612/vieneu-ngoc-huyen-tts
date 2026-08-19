# VieNeu-TTS training và inference

Repo này chứa toàn bộ logic training và inference độc lập với Modal. Modal chỉ cung cấp GPU/Volume; code, config và script đều nằm trong repo để có thể clone và chạy trên local, Kaggle hoặc cloud khác.

## Pipeline

Một lần chạy thực hiện tuần tự trên hai Modal Function độc lập:

1. T4 `16 GB + 4 CPU core / 6 GiB`: mã hóa toàn bộ train/eval bằng NeuCodec và commit cache lên Volume.
2. Sau khi T4 hoàn tất, A10 `24 GB + 4 CPU core / 6 GiB`: Phase 1 LoRA, `lr=1e-4`, tối đa 5 epoch.
3. Trên cùng A10: Phase 2 nạp base BF16 sạch, gắn adapter, `safe_merge`, kiểm tra logits trước/sau merge rồi lưu full model.
4. Trên cùng A10: Phase 3 dùng optimizer mới, mở 33% block trên cùng + output head/final norm, `lr=1e-6`, tối đa 2 epoch.

NeuCodec chỉ chạy trong Function T4, không được load trên Function A10 và không tham gia optimizer. Function T4 kiểm tra CUDA trước khi encode; Function A10 chỉ được gọi sau khi cache có `COMPLETE.json`. NeuCodec vẫn encode từng audio vì implementation hiện tại không có padding mask an toàn cho batch audio khác độ dài. Cả hai phase có gradient đều đánh giá trên `metadata_eval.csv`, early stopping và nạp lại checkpoint có `eval_loss` tốt nhất trước khi xuất artifact cuối.

Thiết lập mặc định cho A10 24 GB dùng effective batch size 32. Phase 3 dùng micro-batch 2, accumulation 16 và gradient checkpointing vì đây là phase tốn VRAM nhất.

## Cấu trúc

```text
configs/
  pipeline_3phase.yaml       # toàn bộ hyperparameter
cloud/
  modal_app.py               # T4 NeuCodec -> shared Volume -> A10 training
scripts/
  train_modal.ps1            # Windows
  train_modal.sh             # Linux/macOS
src/train/
  config.py                  # dataclass + YAML + override CLI
  data.py                    # audit, phoneme, NeuCodec cache, dataset/collator
  modeling.py                # load model, LoRA, unfreeze, save
  phases.py                  # ba phase
  pipeline.py                # orchestration, resume, state
  cli.py                     # chạy core không phụ thuộc Modal
src/inference/
  preprocessing.py           # normalize, chunk, phoneme
  batching.py                # logical batch + phân phối worker
  modeling.py                # full model/LoRA cũ, reference, NeuCodec
  generation.py              # one model replica/GPU, OOM split + retry
  audio.py                   # ghép segment theo index thành một WAV
  api.py                     # API import được, batches=None
  cli.py                     # CLI không phụ thuộc môi trường
notebooks/
  inference_kaggle.ipynb     # hướng dẫn Kaggle không hard-code token
tests/
```

Thư mục `data/` và `outputs/` bị loại khỏi Git bằng `.gitignore`.

Dataset không được đưa vào repo. Dataset hiện tại mang giấy phép CC BY-NC 4.0 theo `data/SOURCE.txt`; cần giữ điều kiện phi thương mại đó tách biệt với giấy phép của code/model.

## Dataset contract

Volume `tts-dataset` hiện được mount tại `/mnt/tts-dataset`; pipeline đọc dataset ở `/mnt/tts-dataset/dataset`:

```text
/dataset/
  audio/*.wav
  metadata.csv          # filename.wav|text
  metadata_eval.csv     # filename.wav|text
  processing_config.json
  processing_stats.json
```

Pipeline kiểm tra file thiếu, filename trùng giữa train/eval và sequence dài quá 2.048 token. Audio được downsample về 16 kHz đúng đầu vào NeuCodec; dữ liệu mã hóa không bị cắt ngẫu nhiên xuống 2.000 mẫu.

Ở lần chạy đầu, hai giá trị revision đang để `null` sẽ được resolve thành commit SHA bất biến của model và codec rồi ghi vào `resolved_config.yaml`. Mọi lần resume dùng lại đúng SHA đó; đổi model/codec hoặc hyperparameter nhưng giữ nguyên run name sẽ bị từ chối để tránh trộn checkpoint.

## Chạy trên Windows

Yêu cầu local: Python 3.12 và `modal==1.5.4`. Máy local không cần cài PyTorch/CUDA.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[cloud]"
```

### Hugging Face token

NeuCodec là gated model. Tài khoản Hugging Face phải được cấp quyền truy cập `neuphonic/neucodec`, sau đó token được giữ trong file local không đưa lên Git:

```powershell
Copy-Item .env.example .env.modal
notepad .env.modal
```

Điền token read-only vào `.env.modal`:

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

Đồng bộ file này vào Modal Secret `huggingface-secret`:

```powershell
.\scripts\sync_modal_secrets.ps1
```

Nếu PowerShell chặn script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_modal_secrets.ps1
```

`.env.modal` khớp quy tắc `.env.*` trong `.gitignore`; code chỉ tham chiếu Modal Secret và không đóng gói file/token vào image.

Chạy cả ba phase bằng một lệnh:

```powershell
.\scripts\train_modal.ps1
```

Nếu PowerShell của máy đang chặn script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train_modal.ps1
```

Có thể override các tham số chính mà không sửa code:

```powershell
.\scripts\train_modal.ps1 `
  -RunName "ngoc-a10-v2" `
  -Phase1LearningRate 0.0001 `
  -Phase1Epochs 5 `
  -Phase3LearningRate 0.000001 `
  -Phase3Epochs 2
```

Hoặc gọi Modal trực tiếp:

```powershell
modal run --detach --env main cloud/modal_app.py `
  --config pipeline_3phase.yaml `
  --run-name ngoc-a10-v1
```

`--detach` bảo đảm job không bị dừng nếu terminal mất kết nối. Lệnh gọi Function T4 trước; chỉ khi Function này thành công mới gọi Function A10. Nếu T4 hoặc A10 lỗi, lần chạy kế tiếp tiếp tục từ cache/checkpoint đã commit. Không có bước mở trình duyệt và không tự tải checkpoint về máy.

## Artifact trên Modal

`tts-training-results` được tạo tự động. Sau khi thành công, mỗi phase chỉ giữ một artifact cuối:

```text
/runs/ngoc-a10-v1/
  resolved_config.yaml
  preparation_environment.json
  environment.json
  pipeline_state.json
  pipeline_summary.json
  phase1_lora/final/       # LoRA adapter tốt nhất
  phase2_merge/final/      # full model đã merge
  phase3_finetune/final/   # full model cuối cùng
```

Checkpoint Trainer tạm nằm trong `.runtime/`, được commit định kỳ để resume khi preempt và xóa sau khi phase đã lưu + commit artifact hoàn chỉnh. Cache NeuCodec được giữ ở `/cache/encoded/` để lần retry/run tương thích không cần encode lại.

Xem artifact và log:

```powershell
modal volume ls --env main tts-training-results /runs/ngoc-a10-v1
modal app logs --env main vieneu-tts-three-phase --follow
```

## Chạy core ngoài Modal

Khi một máy Linux/CUDA đã cài đủ dependency:

```bash
python -m pip install -e ".[train]"
python -m train --config configs/pipeline_3phase.yaml \
  --dataset-root /path/to/data \
  --output-root /path/to/results \
  --run-name local-test
```

Chỉ kiểm tra config và dataset local, không tải model:

```powershell
$env:PYTHONPATH = "src"
python -m train --config configs/pipeline_3phase.yaml --validate-only
```

## Inference độc lập môi trường

Đầu vào model chuẩn là artifact full model cuối Phase 3:

```text
/runs/<run-name>/phase3_finetune/final/
```

Pipeline thực hiện: đọc TXT → normalize giống dữ liệu train → chia đoạn `80/128/156` ký tự → sea-g2p → logical batch `128`, length gap `12` → encode reference đúng một lần → sinh NeuCodec token → decode các segment → sắp lại theo `index` → ghi nguyên tử một file WAV. Nếu một segment vẫn lỗi sau retry, pipeline không tạo WAV thiếu đoạn.

`batch_size=128` là logical batch. `max_runtime_batch_size` có thể chia forward pass nhỏ hơn để vừa VRAM mà không đổi dữ liệu/batching. Khi có nhiều GPU, mỗi GPU giữ một bản đầy đủ của model và xử lý các batch độc lập; đây là data parallel inference, không phải model sharding.

### CLI local/Linux/Kaggle

Máy chạy inference cần cài dependency đầy đủ:

```powershell
python -m pip install -e ".[inference]"
```

Chạy một GPU tự động:

```powershell
python -m inference `
  --config configs/inference.yaml `
  --model "D:\models\ngoc-a10-v1\phase3_finetune\final" `
  --input "D:\stories\story.txt" `
  --reference-audio "D:\Project\TTS\data\audio\ngochuyen_00769.wav" `
  --reference-text "Vì thế, Đồng chí luôn được các Đồng chí lãnh đạo cấp cao của Đảng tin tưởng, đánh giá cao." `
  --output "D:\stories\story_complete.wav"
```

Hai GPU:

```powershell
python -m inference ... --devices "cuda:0,cuda:1" --num-gpus 2
```

Trên T4 có thể giữ logical batch 128 nhưng giới hạn mỗi forward:

```powershell
python -m inference ... --batch-size 128 --max-runtime-batch-size 8
```

### Python API

Không truyền `batches` thì `infer()` tự tiền xử lý và tạo batch:

```python
from inference import InferenceConfig, TTSInference

config = InferenceConfig(
    model="/path/to/phase3_finetune/final",
    num_gpus=1,
)

with TTSInference(config) as tts:
    result = tts.infer(
        input_path="story.txt",
        reference_audio="ngochuyen_00769.wav",
        reference_text="Vì thế, Đồng chí luôn được các Đồng chí lãnh đạo cấp cao của Đảng tin tưởng, đánh giá cao.",
        output_path="story_complete.wav",
    )
```

Muốn xem hoặc sửa batch trước khi infer:

```python
from inference import prepare_batches

batches = prepare_batches(
    input_path="story.txt",
    min_chars=80,
    target_chars=128,
    max_chars=156,
    batch_size=128,
    max_length_gap=12,
)
result = tts.infer(
    batches=batches,
    reference_audio="ngochuyen_00769.wav",
    reference_text="Vì thế, Đồng chí luôn được các Đồng chí lãnh đạo cấp cao của Đảng tin tưởng, đánh giá cao.",
    output_path="story_complete.wav",
)
```

### Modal inference

Upload TXT một lần:

```powershell
.\scripts\upload_inference_text.ps1 -InputPath "D:\stories\story.txt"
```

Chạy trên một A10, đọc model trực tiếp từ `tts-training-results` và chỉ lưu WAV cuối vào `tts-inference-results`:

```powershell
.\scripts\infer_modal.ps1 `
  -Model "/mnt/tts-results/runs/ngoc-a10-v1/phase3_finetune/final" `
  -InputPath "story.txt" `
  -OutputPath "story_complete.wav"
```

Nhiều GPU Modal:

```powershell
.\scripts\infer_modal.ps1 -Gpu "A10" -NumGpus 2
```

Tải WAV về khi cần:

```powershell
modal volume get --env main tts-inference-results /story_complete.wav .\outputs\story_complete.wav
```

Notebook hướng dẫn nằm tại `notebooks/inference_kaggle.ipynb`. Notebook chỉ đọc `HF_TOKEN` và `GITHUB_TOKEN` từ Kaggle Secrets, không chứa token thật.

### Kiểm thử không tải model

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m compileall -q src cloud tests
modal run cloud\modal_inference.py --help
```
