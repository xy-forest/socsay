"""三类传统人脸识别器对比 demo（headless，无 GUI 弹窗）。

OpenCV 的 cv2.face 模块提供：
  - LBPHFaceRecognizer  : 局部二值模式直方图，最常用
  - EigenFaceRecognizer : PCA + 欧氏距离
  - FisherFaceRecognizer: LDA（监督），需要每人 ≥2 张 + 至少 2 类

依赖： opencv-contrib-python（已替换 opencv-python）

用法：
    # 1) 采集（默认抓 Insta360 Link 2，自动找索引）
    python lbph_demo.py collect ink 25
    python lbph_demo.py collect tom 25
    # 2) 训练三种识别器
    python lbph_demo.py train
    # 3) 留出集自评对比
    python lbph_demo.py eval
    # 4) 实时识别若干帧（headless：仅打印 + 落盘预览）
    python lbph_demo.py live 5

输出：
    /tmp/lbph_faces/<label>_<n>.png   采集的脸（128x128 灰度）
    /tmp/lbph_faces/lbph_model.xml
    /tmp/lbph_faces/eigen_model.xml
    /tmp/lbph_faces/fisher_model.xml
    /tmp/lbph_faces/last_capture.jpg  最近一帧带框预览
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path("/tmp/lbph_faces")
DEVICE_NAME = "Insta360 Link 2"
FACE_SIZE = (128, 128)
DETECTOR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _find_avf_index(name: str = DEVICE_NAME) -> int:
    """从 ffmpeg avfoundation 列表中找到设备索引；失败回 0。"""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return 0
    text = (r.stderr or "") + (r.stdout or "")
    for line in text.splitlines():
        if name in line:
            m = re.search(r"\[(\d+)\]\s*" + re.escape(name), line)
            if m:
                return int(m.group(1))
    return 0


def _open_camera() -> cv2.VideoCapture:
    idx = int(os.environ.get("CAM_INDEX", _find_avf_index()))
    print(f"[cam] open AVFoundation index={idx}")
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"camera index {idx} open failed")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def _detect_largest(gray: np.ndarray):
    faces = DETECTOR.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


# --------- 1) collect ---------

def collect(label: str, num: int = 25):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cap = _open_camera()
    saved = 0
    misses = 0
    t0 = time.monotonic()
    while saved < num and time.monotonic() - t0 < 60:
        ok, frame = cap.read()
        if not ok:
            misses += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = _detect_largest(gray)
        if bbox is None:
            misses += 1
            time.sleep(0.05)
            continue
        x, y, w, h = bbox
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, FACE_SIZE)
        face = cv2.equalizeHist(face)
        fp = DATA_DIR / f"{label}_{saved:03d}.png"
        cv2.imwrite(str(fp), face)
        saved += 1
        prev = frame.copy()
        cv2.rectangle(prev, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(prev, f"{label} {saved}/{num}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(DATA_DIR / "last_capture.jpg"), prev)
        print(f"  saved {fp.name}  (face {w}x{h})")
        time.sleep(0.15)
    cap.release()
    print(f"[collect] label={label} saved={saved} misses={misses}")


# --------- 2) train ---------

def _load_dataset():
    files = sorted(DATA_DIR.glob("*.png"))
    if not files:
        raise RuntimeError(f"no faces in {DATA_DIR}, run collect first")
    label2id: dict[str, int] = {}
    images, ids = [], []
    for f in files:
        if f.name == "last_capture.jpg":
            continue
        label = f.stem.split("_")[0]
        if label not in label2id:
            label2id[label] = len(label2id)
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape[:2] != FACE_SIZE:
            img = cv2.resize(img, FACE_SIZE)
        images.append(img)
        ids.append(label2id[label])
    return images, np.array(ids, dtype=np.int32), label2id


def train():
    images, ids, label2id = _load_dataset()
    print(f"[train] {len(images)} samples, classes={label2id}")

    lbph = cv2.face.LBPHFaceRecognizer_create()
    lbph.train(images, ids)
    lbph.save(str(DATA_DIR / "lbph_model.xml"))
    print("  saved lbph_model.xml")

    if len(label2id) < 2:
        print("  [skip] Eigen/Fisher 需要 >=2 类，请再 collect 一个不同 label")
    else:
        eig = cv2.face.EigenFaceRecognizer_create()
        eig.train(images, ids)
        eig.save(str(DATA_DIR / "eigen_model.xml"))
        print("  saved eigen_model.xml")

        fish = cv2.face.FisherFaceRecognizer_create()
        fish.train(images, ids)
        fish.save(str(DATA_DIR / "fisher_model.xml"))
        print("  saved fisher_model.xml")

    (DATA_DIR / "label_map.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2)
    )


# --------- 3) eval ---------

def evaluate():
    images, ids, label2id = _load_dataset()
    print(f"[eval] {len(images)} samples, classes={label2id}")

    recognizers = [("LBPH", cv2.face.LBPHFaceRecognizer_create())]
    if len(label2id) >= 2:
        recognizers.append(("Eigen", cv2.face.EigenFaceRecognizer_create()))
        recognizers.append(("Fisher", cv2.face.FisherFaceRecognizer_create()))
    else:
        print("  [warn] 仅 1 类，Eigen/Fisher 跳过")

    for name, rec in recognizers:
        rec.train(images, ids)
        correct = 0
        confs = []
        for img, gold in zip(images, ids):
            pid, conf = rec.predict(img)
            confs.append(conf)
            if pid == gold:
                correct += 1
        avg = sum(confs) / len(confs) if confs else 0
        print(f"  {name:7s}  acc={correct}/{len(images)} ({correct/len(images):.0%})  "
              f"avg_conf={avg:.1f}  (LBPH:越小越好；Eigen/Fisher:越小越像)")


# --------- 4) live recognize ---------

def live(n_frames: int = 5):
    if not (DATA_DIR / "lbph_model.xml").exists():
        raise RuntimeError("先 train")
    label2id = json.loads((DATA_DIR / "label_map.json").read_text())
    id2label = {v: k for k, v in label2id.items()}

    lbph = cv2.face.LBPHFaceRecognizer_create()
    lbph.read(str(DATA_DIR / "lbph_model.xml"))
    eig = fish = None
    if (DATA_DIR / "eigen_model.xml").exists():
        eig = cv2.face.EigenFaceRecognizer_create()
        eig.read(str(DATA_DIR / "eigen_model.xml"))
    if (DATA_DIR / "fisher_model.xml").exists():
        fish = cv2.face.FisherFaceRecognizer_create()
        fish.read(str(DATA_DIR / "fisher_model.xml"))

    cap = _open_camera()
    seen = 0
    t0 = time.monotonic()
    while seen < n_frames and time.monotonic() - t0 < 30:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox = _detect_largest(gray)
        if bbox is None:
            time.sleep(0.05)
            continue
        x, y, w, h = bbox
        face = cv2.equalizeHist(cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE))
        out = []
        pid, c = lbph.predict(face)
        out.append(f"LBPH={id2label.get(pid, '?')}({c:.1f})")
        if eig is not None:
            pid, c = eig.predict(face)
            out.append(f"Eigen={id2label.get(pid, '?')}({c:.1f})")
        if fish is not None:
            pid, c = fish.predict(face)
            out.append(f"Fisher={id2label.get(pid, '?')}({c:.1f})")
        seen += 1
        print(f"  frame#{seen}: {' | '.join(out)}")
        prev = frame.copy()
        cv2.rectangle(prev, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(prev, " | ".join(out), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(DATA_DIR / "last_recognize.jpg"), prev)
        time.sleep(0.2)
    cap.release()
    print(f"[live] done. preview at {DATA_DIR/'last_recognize.jpg'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "collect":
        label = sys.argv[2] if len(sys.argv) > 2 else "user"
        num = int(sys.argv[3]) if len(sys.argv) > 3 else 25
        collect(label, num)
    elif cmd == "train":
        train()
    elif cmd == "eval":
        evaluate()
    elif cmd == "live":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        live(n)
    else:
        print("命令: collect <label> [n] | train | eval | live [n]")
        sys.exit(2)
