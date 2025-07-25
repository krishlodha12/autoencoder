import os
import socket
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

SOCKET_PATH = "/tmp/dpdk_socket_8node"
FEATURE_KEYS = [
    "_not_ipv4_ipv6", "_arp", "_udp", "_ipv6",
    "_ipv4", "_icmp", "_tcp", "_frame_bin"
]

log_dir = "logs/live"
run_log_dir = os.path.join(log_dir, "run1")
os.makedirs(run_log_dir, exist_ok=True)

for entry in os.listdir(run_log_dir):
    path = os.path.join(run_log_dir, entry)
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        import shutil
        shutil.rmtree(path)

def build_autoencoder(input_dim, latent_dim):
    inputs = Input(shape=(input_dim,), name="input")
    x = Dense(32, activation="relu")(inputs)
    x = Dense(latent_dim, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(input_dim, activation="linear")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

# Build the model before tracing
model = build_autoencoder(input_dim=len(FEATURE_KEYS), latent_dim=8)
scaler = MinMaxScaler()
writer = tf.summary.create_file_writer(run_log_dir)

# Trace model graph for TensorBoard
dummy = tf.convert_to_tensor(np.zeros((1, len(FEATURE_KEYS))), dtype=tf.float32)

@tf.function
def trace_step(x):
    return model(x)

with writer.as_default():
    tf.summary.trace_on(graph=True, profiler=False)
    trace_step(dummy)
    tf.summary.trace_export(name="model_trace", step=0)

# Prepare socket
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
server.listen(1)
print(f"[+] Waiting for connection at {SOCKET_PATH}...")
conn, _ = server.accept()
print("[+] Connected.")

# Training loop
buffer = b""
step = 1
scaler_fitted = False

try:
    while True:
        data = conn.recv(4096)
        if not data:
            break
        buffer += data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            try:
                parsed = json.loads(line.decode())
                print(f"Received JSON: {parsed}")
                row = [parsed.get(k, 0) for k in FEATURE_KEYS]
                batch = np.array([row])

                if not scaler_fitted:
                    scaler.fit(batch)
                    scaler_fitted = True

                scaled = scaler.transform(batch)

                print("Original:", row)
                print("Scaled:", scaled.tolist())

                if not np.allclose(scaled, 0):
                    history = model.fit(scaled, scaled, epochs=1, verbose=0)
                    loss = history.history["loss"][0]
                    print(f"[Step {step}] Loss: {loss:.6f}")

                    with writer.as_default():
                        tf.summary.scalar("loss", loss, step=step)
                        writer.flush()
                    step += 1
                else:
                    print(f"[Step {step}] Skipped: all-zero input")
            except Exception as e:
                print(f"Error processing line: {e}")
except KeyboardInterrupt:
    print("Interrupted.")
finally:
    conn.close()
    server.close()
    model.save("autoencoder_model_8node.h5")
    print("Model saved.")
