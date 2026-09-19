# NEXUS Troubleshooting Guide

---

## Backend Won't Start

### "Address already in use" / Port 8000 Occupied

```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process by PID (replace 1234 with actual PID)
taskkill /PID 1234 /F

# Or use a different port
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

### Import Error on Startup

```
ModuleNotFoundError: No module named 'backend'
```

**Cause**: Running uvicorn from the wrong directory.  
**Fix**: Always run from the project root (`nexus/`), not from `backend/`:

```bash
cd C:\path\to\nexus
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Database Initialization Error

```
sqlite3.OperationalError: unable to open database file
```

**Cause**: `data/` directory doesn't exist.  
**Fix**:
```bash
mkdir -p data/documents data/reports data/audio
```

---

## UI Issues

### Page Won't Load / 404 Error

**Cause**: Frontend files not found.  
**Check**: The `frontend/` directory must exist at the project root.  
**Fix**: Confirm you cloned the full repository including `frontend/`.

### NPU Shows "CHECKING" and Never Updates

**Cause**: The health endpoint is unreachable.  
**Fix**: Confirm the backend is running and responding:
```bash
curl http://127.0.0.1:8000/health
```

### Voice Input Button Doesn't Work

**Cause**: Web Speech API requires a Chromium-based browser (Chrome or Edge).  
**Fix**: Open the UI in Microsoft Edge or Google Chrome, not Firefox or Safari.  
**Alternative**: Type your goal in the text field instead.

---

## Model Loading Issues

### "Model file not found" Warning

NEXUS runs without downloaded model files — components report `NOT_IMPLEMENTED` honestly. This is expected behavior on a fresh installation.

**To enable a component**, download the model file (see [`MODEL_SETUP.md`](MODEL_SETUP.md)).

### ONNX Runtime Version Mismatch

```
OnnxRuntimeError: ONNXRuntime version mismatch
```

**Fix**: Ensure you're using ONNX Runtime 1.18+ with matching model ONNX opset versions:
```bash
pip install --upgrade onnxruntime
```

### LLM Response Timeout

```
AgentError: LLM call timed out after 60 seconds
```

**Cause**: The local LLM is running on CPU and is slower than the 60-second timeout.  
**Fix**:
1. On Snapdragon: Ensure `QNNExecutionProvider` is active for ~3x speedup
2. On x64 development host: Accept CPU inference latency (normal for CPU-only runs)
3. Increase timeout in `backend/config.py`: `LLM_TIMEOUT_SECONDS = 120`

---

## Agent / Planning Issues

### "Failed to parse plan" Error

**Cause**: The LLM produced malformed JSON (not properly formatted structured plan).  
**What Happens**: NEXUS reports this cleanly as a planning error — not a crash.  
**Fix**: Retry the task. With CPU inference, occasional malformed outputs are possible. The system will retry once automatically.

**Non-technical explanation**: The AI's response didn't follow the expected format. Please try again — this usually succeeds on the second attempt.

### "Cyclic dependency detected in plan"

**Cause**: The planner created a task dependency loop (A depends on B depends on A).  
**What Happens**: Kahn's algorithm catches this and rejects the plan before any tasks run.  
**Fix**: Retry with a more specific goal description.

### Tasks Running But No Results Appearing

**Check task status**:
```bash
curl http://127.0.0.1:8000/api/agent/tasks
```

Look for `status: "running"` — if tasks are stuck, check the backend logs for errors.

---

## Document Ingestion Issues

### "File type not supported"

**Supported types**: `.pdf`, `.docx`, `.txt`, `.png`, `.jpg`, `.jpeg`, `.json`  
**Fix**: Convert your file to a supported format.

### "File size exceeds 50MB limit"

**Cause**: The hard size ceiling is 50MB per file.  
**Fix**: Split large documents into smaller sections.

### "File validation failed: magic byte mismatch"

**Cause**: The file extension doesn't match the actual file content (e.g., an `.exe` renamed to `.pdf`).  
**Fix**: Ensure the file is a genuine document of the declared type.

### PDF Text Extraction Returns Empty

**Cause**: The PDF is a scanned image (no embedded text layer).  
**Fix**: Use the image upload route instead, which routes through the OCR/vision pipeline.

---

## Demo Issues

### demo.py Exits Immediately

**Check**: Is the backend running?
```bash
curl http://127.0.0.1:8000/health
```
If not, start it first in a separate terminal.

### Demo Data Missing

```
FileNotFoundError: demo_data/ not found
```

**Fix**:
```bash
python scripts/generate_demo_data.py
```

### Demo Shows 10/10 But Report Is Empty

**Cause**: Report generation step encountered an error.  
**Fix**: Check `data/reports/` directory for partially generated files. Check backend logs for the specific error.

---

## Performance Issues

### Very Slow Response (Minutes per Request)

On a CPU-only development host, LLM token generation is slow (~700ms for 16 tokens). A full agent workflow may take 15–60 seconds.

**This is expected behavior on CPU.** On Snapdragon X Elite with NPU:
- LLM: ~40–48 tokens/second (significantly faster)
- Embedding: ~15ms per chunk
- Vision: ~35ms per image

### High Memory Usage

Full pipeline with all models loaded: ~2.5 GB RSS on CPU.

**Fix**: 
- Close other memory-intensive applications
- Use `uvicorn --workers 1` to prevent duplicate model instances

---

## Test Suite Issues

### Tests Failing After Code Changes

```bash
# Run tests with verbose output to see exact failures
pytest tests/ -v --tb=short

# Run only specific test file
pytest tests/test_security.py -v

# Run only passing tests from before
pytest tests/ -k "not test_new_feature" -v
```

### Import Errors in Tests

**Cause**: Test environment not using the project's virtual environment.  
**Fix**:
```bash
.venv\Scripts\Activate.ps1
pytest tests/ -v
```

---

## Getting More Help

1. Check the backend logs in your terminal for detailed error messages
2. Enable debug logging: `NEXUS_LOG_LEVEL=debug python -m uvicorn backend.main:app ...`
3. Review [`docs/LIMITATIONS.md`](../LIMITATIONS.md) for known constraints
4. Check the API documentation at `http://127.0.0.1:8000/docs`
