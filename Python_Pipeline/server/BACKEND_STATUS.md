# Python Backend - Test & Status

## Durum: ✅ HAZIR

### Dosyalar Kontrol Edildi:
- ✅ `Pipeline.py` - Yok
- ✅ `pipeline_http_bridge.py` - Sorun yok
- ✅ `audio_buffer.py` - Sorun yok
- ✅ `load_env.py` - Sorun yok
- ✅ `requirements.txt` - Sorun yok
- ✅ `.env` - Gemini API key ayarlanmış

### Yapılan Düzeltmeler:
1. **HTTP Host Binding**: Default host `172.20.10.2` → `0.0.0.0` (tüm interfaces)
   - Ağ konfigürasyonundan bağımsız çalışabilir
   - Android cihaz veya localhost'tan erişilebilir

### Başlatma Komutu:

```bash
cd /Users/mesely/449_pitonf
source .venv/bin/activate

# 0.0.0.0:8000 üzerinde başlat
PIPELINE_HTTP_HOST=0.0.0.0 PIPELINE_HTTP_PORT=8000 python Pipeline.py

# VEYA custom IP ile (Android cihazın IP'sini biliyorsan)
PIPELINE_HTTP_HOST=172.20.10.2 PIPELINE_HTTP_PORT=8000 python Pipeline.py
```

### HTTP Server Endpoints:

```
POST /client_hello
  Body: {"session_id": "device_id"}
  Response: {"status": "ok"}

POST /audio_chunk
  Body: {
    "session_id": "device_id",
    "seq": 1,
    "timestamp_unix": 1234567890.123,
    "pcm_base64": "base64_encoded_float32_pcm"
  }
  Response: {"status": "received"}

GET /events?session_id=device_id&since_unix=0.0
  Response: {
    "events": [...],
    "last_timestamp_unix": 1234567890.123
  }
```

---

## Unity Tarafında Düzeltmesi Gerekenler:

### 1. ⚠️ Non-secure HTTP Bağlantısı Hatası
**Error**: `InvalidOperationException: Insecure connection not allowed`

**Çözüm**: Unity Player Settings'te insecure HTTP izni ver:

```
Edit → Project Settings → Player → Settings for Android/iOS
→ Other Settings
→ Security
→ Insecure HTTP Client Downloads: AlwaysAllowed
```

### 2. ⚠️ ADB Device Offline
**Error**: `adb: device offline`

**Çözüm**:
```bash
# Terminal'de:
adb kill-server
adb start-server
adb connect 172.20.10.7:5555  # Senin cihazın IP'si
adb devices  # Kontrol et
```

### 3. ⚠️ Microphone Izni
**Error**: `[AudioCaptureController] No microphone devices found`

**Çözüm**: Unity Player Settings:
```
Edit → Project Settings → Player
→ Android: Add microphone permission
→ iOS: Set NSMicrophoneUsageDescription
```

### 4. ✅ BackendHttpClient Ayarları

`Assets/Scripts/Networking/BackendHttpClient.cs` dosyasında kontrol et:

```csharp
using UnityEngine.Networking;

void Start()
{
    // HTTP_BASE_URL'yi kendi Python server'ına ayarla
    string serverUrl = "http://0.0.0.0:8000";  // veya IP
    
    StartCoroutine(SendClientHello(serverUrl));
}

IEnumerator SendClientHello(string baseUrl)
{
    using (UnityWebRequest request = UnityWebRequest.PostJson(
        baseUrl + "/client_hello",
        new { session_id = "my_device" }
    ))
    {
        yield return request.SendWebRequest();
        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Client hello failed: " + request.error);
        }
    }
}
```

---

## Basit Test Akışı:

1. Python pipeline'ı başlat:
   ```bash
   PIPELINE_HTTP_HOST=0.0.0.0 PIPELINE_HTTP_PORT=8000 python Pipeline.py
   ```

2. Unity uygulamayı Android cihazda çalıştır

3. BackendHttpClient otomatik olarak `/client_hello` ve `/audio_chunk` gönderecek

4. Python tarafı `/events` endpoint'ten sorgulanabilir

---

## İpuçları:

- **Network IP**: Android cihazdan `adb shell ip addr` ile cihazın IP'sini öğren
- **Port forwarding**: `adb forward tcp:8000 tcp:8000`
- **Logs**: Python ve Unity debug logs'unu açık tut
- **Firewall**: macOS firewall'ın 8000 portunu engellemiyor mu kontrol et
