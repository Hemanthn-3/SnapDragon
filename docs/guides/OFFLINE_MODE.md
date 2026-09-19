# NEXUS Offline Mode Guide

NEXUS is designed to be **100% offline by default**. This guide explains how the air-gap works, how to verify it, and how to configure it for different deployment scenarios.

---

## Default Behaviour

Out of the box, NEXUS:
- Sets `OFFLINE_MODE = True` in `backend/config.py`
- Activates `LocalNetworkGuard` at application startup
- Blocks all outbound socket connections to non-loopback addresses
- Logs all blocked attempts to the in-memory audit trail

You do not need to do anything to enable offline mode — it is the default.

---

## How the Air-Gap Works

### Python Socket Layer Hook

`backend/network/guard.py` patches the Python socket module at import time:

```python
# Simplified representation
original_connect = socket.socket.connect

def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else str(address)
    if not _is_loopback(host):
        # Log the blocked attempt
        _audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "blocked_host": host,
            "reason": "StrictLocalOnlyViolationError"
        })
        raise StrictLocalOnlyViolationError(
            f"Outbound connection to '{host}' blocked by NEXUS LocalNetworkGuard. "
            f"NEXUS operates in strict offline-only mode."
        )
    return original_connect(self, address)

socket.socket.connect = _guarded_connect
```

This intercepts connections from:
- `urllib` / `urllib3`
- `requests` / `httpx` / `aiohttp`
- `boto3`, `google-cloud-*`, and any other HTTP library that uses Python's socket module

### Loopback Allowlist

The following addresses are always permitted:
- `127.0.0.1` (IPv4 loopback)
- `::1` (IPv6 loopback)
- `localhost`

---

## Verifying Offline Mode

### Via the Health Endpoint

```bash
curl http://127.0.0.1:8000/health
```

Look for:
```json
{
  "offline_mode": true,
  "network_guard": "active",
  "blocked_attempts": 0
}
```

### Via the Network Status Endpoint

```bash
curl http://127.0.0.1:8000/api/network/status
```

### Via the UI

The Home screen and Agent Workspace both display:

```
LOCAL MODE ● ACTIVE
```

Green indicator = offline mode active, zero network calls.

### Via the Audit Log

```bash
curl http://127.0.0.1:8000/api/network/audit
```

This returns a list of all blocked connection attempts during the current session. On a clean offline run, this list is empty.

---

## Testing the Air-Gap

Run the security test:

```bash
pytest tests/test_security.py::test_network_access_attempts -v
```

This test verifies that attempting `socket.connect("8.8.8.8", 53)` raises `StrictLocalOnlyViolationError`.

---

## Disabling Offline Mode (Development Only)

> [!CAUTION]
> Disabling offline mode allows network connections. Only do this in a development environment where you understand the implications.

Set the environment variable before starting:

```powershell
$env:NEXUS_OFFLINE_MODE = "false"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Or modify `backend/config.py` (not recommended for production):

```python
OFFLINE_MODE: bool = False  # ← Change to False
```

---

## Supplementary OS-Level Air-Gap

For the highest security deployments, supplement Python-layer enforcement with Windows Firewall rules:

```powershell
# Block all outbound from the Python process (Run as Administrator)
New-NetFirewallRule `
  -DisplayName "NEXUS Outbound Block" `
  -Direction Outbound `
  -Program "C:\path\to\nexus\.venv\Scripts\python.exe" `
  -Action Block `
  -Profile Any
```

To allow loopback (for the browser to reach the server):
```powershell
New-NetFirewallRule `
  -DisplayName "NEXUS Loopback Allow" `
  -Direction Outbound `
  -Program "C:\path\to\nexus\.venv\Scripts\python.exe" `
  -RemoteAddress 127.0.0.1 `
  -Action Allow `
  -Profile Any
```

---

## What "Offline" Means for Model Inference

NEXUS models are loaded from local disk at startup:
- No model weights are downloaded at runtime
- No telemetry is sent to Hugging Face, OpenAI, or any cloud provider
- No API keys are used or required
- The ONNX Runtime does not phone home (all telemetry is disabled)

To disable ONNX Runtime telemetry explicitly:

```python
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.enable_profiling = False
# ONNX Runtime telemetry is off by default in offline builds
```

---

## Offline Mode Limitations

- Browser-based voice input (Web Speech API) may attempt to contact Google's speech services in some browsers. Use Firefox or disable web-based speech recognition; NEXUS's own `/api/speech/transcribe` endpoint handles transcription locally via Whisper.
- Windows Update and other OS background services are outside NEXUS's control. For a true air-gap environment, configure Windows Update to manual and disable background app network access in Windows Settings.
