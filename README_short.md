# ECE 341X Final Project — Visual Wake Words on Raspberry Pi

## Model

* Architecture: **MobileNetV1-based Visual Wake Words classifier**
* Task: Binary classification (`person` / `non_person`)
* Deployment format: **TensorFlow Lite**
* Compression: **Post-training FP16 quantization**

## Compression Strategy

The trained Keras model was converted to TensorFlow Lite using **post-training float16 quantization**.
This reduces model size and memory bandwidth while maintaining accuracy.

Conversion settings used:

* `converter.optimizations = [tf.lite.Optimize.DEFAULT]`
* `converter.target_spec.supported_types = [tf.float16]`

This produced a compact deployable model suitable for CPU-only execution on Raspberry Pi.

---

## Final Raspberry Pi Results

Evaluation was performed using the official `scoreboard.py` script with:

* **1 thread**
* **50 warmup images**
* **test_public split**

| Metric          | Result              |
| --------------- | ------------------- |
| Accuracy        | **82.87%**          |
| Model Size      | **0.4371 MB**       |
| MACs            | **7.49 M**          |
| Latency (p90)   | **3.22 ms / image** |
| Peak RSS Memory | **52.42 MB**        |
| Score           | **0.9290**          |

The model exceeds the project requirement of **≥ 80% accuracy** while maintaining very low latency and a compact model size.

---

## Reproducibility Steps

### 1. Train model on cluster GPU

```
python src/train_vww.py
```

### 2. Convert trained model to TensorFlow Lite

```
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

### 3. Generate verified metadata JSON

```
python src/evaluate_vww.py \
  --model models/model.tflite \
  --split test_public \
  --compute_score \
  --export_json
```

### 4. Run Raspberry Pi evaluation

```
python src/scoreboard.py --model models/model.tflite --split test_public --compute_score
```

---

## Environment Notes

Evaluation on the Raspberry Pi was performed inside a lightweight Docker container
(`python:3.11-slim`) to ensure a clean Python environment and avoid dependency
issues when installing `tflite-runtime` on ARM systems.

The container mounts the project directory so the evaluation scripts and dataset
can run directly on the Raspberry Pi filesystem.

Example command used:

docker run --rm -it \
  -v "$PWD:/work" \
  -w /work \
  python:3.11-slim bash


## Local Results (test_public)
- Accuracy: 82.87%
- Model Size: 0.4371 MB
- MACs: 7.4897 M
- Latency (p90, 1 thread): 3.82 ms
- Peak RSS: 359.50 MB


## Final results (Raspberry Pi, threads=1, test_public)
- Model: model.tflite
- Accuracy: 0.8287 (82.87%)
- Model size: 0.4371 MB
- MACs: 7.4897 M
- Latency (official p90): 3.22 ms
- Peak RSS memory: 52.42 MB
- Score: 0.9290
