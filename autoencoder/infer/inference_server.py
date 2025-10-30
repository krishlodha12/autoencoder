import os
import json
import socket
import threading
import numpy as np
import tensorflow as tf

FEATURE_KEYS = [
    "_not_ipv4_ipv6", "_arp", "_udp", "_ipv6",
    "_ipv4", "_icmp", "_tcp", "_frame_bin"
]

MODEL_PATH = os.getenv("MODEL_PATH", "/models/autoencoder_model.keras")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "6006"))

FIXED_MIN = np.zeros(len(FEATURE_KEYS), dtype=np.float32)
FIXED_MAX = np.ones(len(FEATURE_KEYS), dtype=np.float32)

def scale_batch(x):
    # Safe min-max scaling with fallback to identity if min==max
    denom = (FIXED_MAX - FIXED_MIN)
    denom[denom == 0.0] = 1.0
    return (x - FIXED_MIN) / denom

def mse(a, b):
    return float(np.mean(np.square(a - b)))

def handle_client(conn, addr, model):
    buf = b""
    step = 0
    print(f"[inference] client connected from {addr}")
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8", errors="ignore"))
                    # Build feature vector
                    row = [obj.get(k, 0) for k in FEATURE_KEYS]
                    x = np.array([row], dtype=np.float32)
                    x_scaled = scale_batch(x)
                    y = model(x_scaled, training=False).numpy()
                    err = mse(x_scaled, y)
                    step += 1
                    # Print one JSON per capture to stdout for kubectl logs
                    print(json.dumps({"step": step, "error": err, "features": obj}), flush=True)
                except Exception as e:
                    print(json.dumps({"parse_error": str(e)}), flush=True)
    finally:
        conn.close()
        print(f"[inference] client disconnected {addr}")

def main():
    print(f"[inference] loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[inference] model loaded")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[inference] listening on {HOST}:{PORT}")

    try:
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr, model), daemon=True)
            t.start()
    except KeyboardInterrupt:
        pass
    finally:
        s.close()

if __name__ == "__main__":
    main()
