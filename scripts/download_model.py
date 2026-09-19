import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/model.onnx.data"
DEST = r"models\llm\Llama-3.2-1B-Instruct\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4\model.onnx.data"
TOTAL_BYTES = 1856839680
NUM_WORKERS = 8
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB chunks

os.makedirs(os.path.dirname(DEST), exist_ok=True)

def download_range(start, end, file_handle, lock):
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(5):
        try:
            resp = requests.get(URL, headers=headers, timeout=30)
            if resp.status_code in (200, 206):
                with lock:
                    file_handle.seek(start)
                    file_handle.write(resp.content)
                return len(resp.content)
        except Exception as e:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download bytes {start}-{end}")

def main():
    print(f"Target: {DEST}")
    print(f"Total size: {TOTAL_BYTES / (1024*1024):.2f} MB across {NUM_WORKERS} workers")

    import threading
    lock = threading.Lock()

    mode = "r+b" if os.path.exists(DEST) else "wb"
    with open(DEST, mode) as f:
        if mode == "wb":
            f.truncate(TOTAL_BYTES)

        chunks = []
        curr = 0
        while curr < TOTAL_BYTES:
            end = min(curr + CHUNK_SIZE - 1, TOTAL_BYTES - 1)
            # Check if this chunk is already written
            f.seek(curr)
            head_bytes = f.read(1024)
            if not any(b != 0 for b in head_bytes):
                chunks.append((curr, end))
            curr = end + 1

        print(f"Remaining chunks to download: {len(chunks)} of 178")
        if not chunks:
            print("All chunks already downloaded!")
            return

        t0 = time.time()
        completed_bytes = (178 - len(chunks)) * CHUNK_SIZE
        last_log = t0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(download_range, c[0], c[1], f, lock) for c in chunks]
            for i, fut in enumerate(futures):
                b = fut.result()
                completed_bytes += b
                now = time.time()
                if now - last_log >= 5 or completed_bytes >= TOTAL_BYTES:
                    elapsed = now - t0
                    print(f"[{completed_bytes / (1024*1024):.1f} / {TOTAL_BYTES / (1024*1024):.1f} MB] downloaded")
                    last_log = now

    t1 = time.time()
    print(f"Download complete in {t1 - t0:.1f}s!")

if __name__ == "__main__":
    main()
