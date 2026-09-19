# NEXUS Windows 11 Setup Guide

This guide covers Windows-specific configuration required to run NEXUS optimally on Windows 11.

---

## 1. PowerShell Execution Policy

By default, Windows may block PowerShell scripts. Allow local scripts:

```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 2. Python Installation

### Option A: Microsoft Store (Simplest)

1. Open Microsoft Store
2. Search for "Python 3.11"
3. Install — this installs to `%LOCALAPPDATA%\Programs\Python\Python311\`

### Option B: Official Installer (Recommended for ARM64)

1. Go to https://python.org/downloads/
2. Download the **Windows ARM64 installer** for Python 3.11
3. Run the installer; check **"Add Python to PATH"**
4. Verify: `python --version` → `Python 3.11.x`

> [!IMPORTANT]
> On Snapdragon, always use the ARM64 native Python build. An x64 emulated Python will not load ARM64 ONNX Runtime wheels correctly.

---

## 3. Visual C++ Redistributables

Some Python packages (onnxruntime, Pillow) require Visual C++ runtime:

1. Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe (x64) or the ARM64 equivalent
2. Install and restart if prompted

---

## 4. Windows Defender Exclusions (Optional)

If Windows Defender slows down model loading or file operations, add an exclusion for the NEXUS directory:

```powershell
# Run as Administrator
Add-MpPreference -ExclusionPath "C:\path\to\nexus"
```

> [!WARNING]
> Only do this if you trust the NEXUS codebase. Review the source code before adding exclusions.

---

## 5. Firewall Configuration (Air-Gap Mode)

For a true air-gapped deployment, configure Windows Firewall to block all outbound connections from the Python process:

```powershell
# Block outbound connections for python.exe (Run as Administrator)
New-NetFirewallRule -DisplayName "NEXUS Python Outbound Block" `
  -Direction Outbound `
  -Program "C:\path\to\.venv\Scripts\python.exe" `
  -Action Block
```

NEXUS's `LocalNetworkGuard` also enforces this at the Python layer — the firewall rule provides an additional OS-level guarantee.

---

## 6. Port Configuration

NEXUS runs on `127.0.0.1:8000` by default. If port 8000 is occupied:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

Check what's using port 8000:
```powershell
netstat -ano | findstr :8000
```

---

## 7. Long Path Support (Optional)

For very deep directory structures, enable long path support:

```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

## 8. Browser for the UI

NEXUS UI works best in:
- Microsoft Edge (Chromium) — Recommended on Windows
- Google Chrome
- Mozilla Firefox

The Web Speech API (voice input) requires a Chromium-based browser.

---

## 9. Running at Startup (Optional)

Create a Task Scheduler task to start NEXUS at login:

```powershell
$action = New-ScheduledTaskAction -Execute "python" `
  -Argument "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000" `
  -WorkingDirectory "C:\path\to\nexus"

$trigger = New-ScheduledTaskTrigger -AtLogon

Register-ScheduledTask -Action $action -Trigger $trigger `
  -TaskName "NEXUS Backend" -RunLevel Highest
```

---

## Verification

After setup, confirm everything works:

```powershell
# In NEXUS directory
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` — you should see the NEXUS home screen with LOCAL MODE active and NPU status displayed.
