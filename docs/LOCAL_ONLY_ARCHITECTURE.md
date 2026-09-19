# NEXUS: Strict Local-Only Architecture & Network Detection Specification

**Project**: NEXUS — Offline-First Multimodal Work Agent  
**Enforcement Policy**: Strict Air-Gap Isolation (Zero Cloud APIs, Zero Remote Telemetry, Zero Remote Models)  
**Security Baseline**: Application-Layer Socket Guard & Operating System Adapter Telemetry  
**Verification Principle**: Verifiable, empirical network request accounting. Zero fabricated metrics.

---

## 1. Architectural Philosophy: The Air-Gapped Work Agent

Traditional "offline" applications often compromise security through background telemetry, license phone-home checks, or remote fallback endpoints. NEXUS eliminates these vulnerabilities by enforcing a hardware- and software-level air gap:

```
                                  NEXUS Application Process
                                              │
            ┌─────────────────────────────────┴─────────────────────────────────┐
            ▼                                                                   ▼
┌───────────────────────────────┐                               ┌───────────────────────────────┐
│     Local IPC & UI Server     │                               │      Outbound Network Call    │
│  (FastAPI on 127.0.0.1:8000)  │                               │  (External IP / Cloud Domain) │
└───────────────┬───────────────┘                               └───────────────┬───────────────┘
                │                                                               │
                ▼                                                               ▼
    ┌───────────────────────┐                                       ┌───────────────────────┐
    │  LocalNetworkGuard    │                                       │   LocalNetworkGuard   │
    │  - Is Loopback? YES   │                                       │   - Is Loopback? NO   │
    └───────────┬───────────┘                                       └───────────┬───────────┘
                │                                                               │
                ▼                                                               ▼
     ALLOW: Increment Loopback Counter                              BLOCK: Raise StrictLocalOnlyViolationError
     (Browser ↔ Local Backend)                                      (Increment Blocked Counter & Audit Log)
```

---

## 2. Network Request Detection & Interception Mechanism

### 2.1 Socket-Level Interception Hook
Rather than relying on environment variables or high-level library flags (which can be bypassed by third-party dependencies), NEXUS intercepts outbound traffic at the C-extension socket boundary:

1. **`socket.socket.connect(sock_self, address)`**:
   - Every TCP/UDP socket connection initiated by any Python module (including `urllib`, `requests`, `httpx`, `aiohttp`, or ONNX telemetry routines) passes through `LocalNetworkGuard._guarded_connect`.
2. **`socket.create_connection(address, ...)`**:
   - High-level socket builder hooked by `LocalNetworkGuard._guarded_create_connection`.
3. **Loopback Whitelist Evaluation**:
   ```python
   LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
   ```
   - If the target host matches `LOOPBACK_HOSTS` or begins with `127.`, the connection is permitted, and the atomic counter `loopback_served` is incremented.
   - If the target host represents any external network IP or remote hostname (e.g., `8.8.8.8`, `api.openai.com`, `telemetry.qualcomm.com`), the connection is **immediately rejected with `StrictLocalOnlyViolationError`**, and the atomic counter `external_blocked` is incremented.

---

## 3. Network Status Detection (Zero Cloud Pings)

Many systems determine network connectivity by pinging public DNS servers (e.g., `8.8.8.8` or `1.1.1.1`). This is unacceptable for an air-gapped agent because sending ping packets leaks metadata.

### 3.1 Kernel Network Adapter Auditing
NEXUS detects network status completely locally using kernel-level adapter enumeration via `psutil`:

1. **`psutil.net_if_stats()`**:
   - Queries the OS kernel for all registered network interface cards (NICs: Wi-Fi, Ethernet, Cellular, Bluetooth, Loopback).
   - Reads the physical `stat.isup` flag, link operational state, and link speed in Mbps.
2. **`psutil.net_if_addrs()`**:
   - Enumerates assigned `AF_INET` (IPv4) and `AF_INET6` (IPv6) addresses without transmitting a single packet.
3. **Status Determination**:
   - If no non-loopback adapter is up, or if `simulated_offline` is engaged, NEXUS reports `Internet: OFFLINE`.
   - If a physical NIC is up, but `LocalNetworkGuard` is active, NEXUS reports `Internet: OFFLINE (AIR-GAPPED)`, guaranteeing that while the host laptop may possess a physical link, NEXUS maintains complete software isolation.

---

## 4. Empirical Network Request Counting

NEXUS does not fake or estimate network numbers. The statistics exposed in the UI and API reflect real, measured occurrences:

| Metric | Source of Truth | Meaning |
| :--- | :--- | :--- |
| **`loopback_served`** | Atomic counter in `LocalNetworkGuard` | Exact number of valid HTTP/REST API calls between the browser UI and the local FastAPI server (`127.0.0.1:8000`). |
| **`external_blocked`** | Atomic counter in `LocalNetworkGuard` | Exact number of outbound attempts to connect to remote IP addresses or cloud services intercepted and blocked. |
| **`audit_log`** | In-memory ring buffer (last 100 entries) | Timestamped audit records of blocked connection attempts with target host and port. |

---

## 5. Offline Workflow Preservation

The core agent workflow operates seamlessly under complete network isolation:

1. **Document Ingestion**: Local file parsing (`pypdf`, `python-docx`, text loaders) requires zero cloud extraction APIs.
2. **Knowledge Retrieval**: Dense vector embeddings (`all-MiniLM-L6-v2`) and cosine similarity search run 100% locally via ONNX Runtime and SQLite WAL.
3. **Small Language Model**: `Llama-3.2-1B-Instruct` executes offline via ONNX Runtime GenAI without cloud LLM endpoints.
4. **Autonomous Planner & Agent Executor**: Topologically sorted task graph and tool execution run entirely within the process sandbox.
5. **Report Generation**: Markdown synthesis and sandboxed disk exports execute locally without remote services.
