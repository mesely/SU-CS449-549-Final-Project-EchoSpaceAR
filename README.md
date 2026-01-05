# 🎵 Real-time Audio Pipeline - START HERE

## ⚡ Quick Start (2 minutes)

### Step 1: Install Python packages
```bash
cd /Users/mesely/ses_yonu_test_2d
bash setup.sh
```

### Step 2: Enable macOS microphone permission
```
System Settings 
  → Privacy & Security 
  → Microphone 
  → Enable for Terminal (or your Python IDE)
```

### Step 3: Run the live audio visualizer
```bash
python3 SimplePipeline.py
```

You should see **real-time audio visualization** with:
- 📊 **SPL graph** (top) - shows sound intensity in dBFS
- 📈 **Frequency spectrum** (bottom) - shows which frequencies are present
- 🎤 **Automatic mic selection** - picks best available microphone

---

## 🎯 What Was Fixed

| Problem | Solution |
|---------|----------|
| ❌ `No microphone found` | ✅ Enhanced mic detection + macOS permissions guide |
| ❌ `HTTP insecure error` | ✅ Enabled via `EnableInsecureHttpDev.cs` |
| ❌ `Android offline` | ✅ Falls back to MacBook Pro mic |
| ❌ `Pipeline missing` | ✅ Created complete HTTP bridge + utilities |

---

## 📂 New Files

- ✅ **`SimplePipeline.py`** ← Start with this!
- ✅ `pipeline_http_bridge.py` - HTTP server for Unity
- ✅ `.env` - Configuration
- ✅ `SETUP.md` - Detailed guide
- ✅ `FIXES.md` - What was fixed
- ✅ `setup.sh` - Auto-install

---

## 🔌 Unity Integration (Optional)

To connect Unity and see audio in Python:

1. **In Python terminal:**
   ```bash
   python3 RealTimeSPLVisualizer.py
   ```
   
2. **In Unity Editor:**
   - Run the scene with `AudioCaptureController.cs`
   - Or run `StereoMicAndroidTest.cs` for Android
   - Watch Unity logs for: `✅ client_hello OK`

3. **Watch Python window** - plots update with Unity audio

---

## 🔧 Configuration

### Python Server (automatically reads `.env`)
```
PIPELINE_HTTP_HOST=0.0.0.0  (listen on all interfaces)
PIPELINE_HTTP_PORT=8000     (port number)
GEMINI_API_KEY=...          (LLM API key)
```

### Unity Client (`BackendHttpClient.cs`)
```csharp
baseUrl = "http://172.20.10.2:8000";  // points to Python server
```

Change to `http://localhost:8000` for local testing.

---

## ✅ Verification Checklist

- [ ] Run `python3 SimplePipeline.py` 
- [ ] See device selection menu
- [ ] See SPL + spectrum plots updating
- [ ] Speak or play sound → see plots spike
- [ ] Check `logs/` folder for CSV files

---

## 📖 More Info

- **Full setup guide:** See `SETUP.md`
- **What was fixed:** See `FIXES.md`
- **Troubleshooting:** See `SETUP.md#Troubleshooting`

---

**Next action:** Run `python3 SimplePipeline.py` and watch the live audio! 🚀

## Reserved folders
- `Android/` — placeholder for Android-side files (to be added by teammates).
- `Python_Pipeline/` — placeholder for Python pipeline/tooling (to be added by teammates).
