using UnityEngine;
using System;
using System.Runtime.InteropServices;
using FMOD;
using FMODUnity;

#if UNITY_ANDROID
using UnityEngine.Android;
#endif

public class StereoMicAndroidTest : MonoBehaviour
{
    [Header("Device Selection")]
    public string preferredDeviceKeyword = "Wireless"; // Rode Wireless vb.
    public bool preferConnectedDrivers = true;         // connected driver varsa onu seç
    public bool fallbackToFirstDevice = true;          // keyword bulamazsa 0. cihaz

    [Header("Recording")]
    public int sampleRate = 48000;
    public float probeIntervalSec = 0.25f;
    public int ringSeconds = 10;                       // ring buffer (sn)
    public uint windowFrames = 2048;                   // RMS için son N frame

    [Header("UI")]
    public bool showOnScreen = true;

    // FMOD
    private FMOD.System fmod;
    private Sound recSound;
    private int recDeviceId = -1;
    private bool recording = false;

    // Selected device info (cache)
    private string recDeviceName = "n/a";
    private int recDeviceChannels = -1;                // driver-reported
    private int recDeviceSysRate = -1;
    private SPEAKERMODE recDeviceMode = SPEAKERMODE.DEFAULT;
    private DRIVER_STATE recDeviceState = 0;

    // Actual capture format we create for the record sound
    private uint actualChannels = 1;
    private const uint BytesPerSample = 2;             // PCM16
    private uint FrameBytes => actualChannels * BytesPerSample;
    private uint ringBytes;

    // Probe results
    private double lastRmsL, lastRmsR;
    private string lastDominant = "n/a";
    private string status = "init";

    // --- Logging helpers: EVERY log has channel count + mic name ---
    private string LogPrefix => $"[FMOD][dev={recDeviceId} name='{recDeviceName}' ch={recDeviceChannels}] ";

    private void Log(string msg) => UnityEngine.Debug.Log(LogPrefix + msg);
    private void LogW(string msg) => UnityEngine.Debug.LogWarning(LogPrefix + msg);
    private void LogE(string msg) => UnityEngine.Debug.LogError(LogPrefix + msg);

    void Start()
    {
        Application.runInBackground = true;

#if UNITY_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            status = "Requesting MICROPHONE permission...";
            UnityEngine.Debug.Log(LogPrefix + status);
            Permission.RequestUserPermission(Permission.Microphone);
            return; // izin gelince Update içinde devam
        }
#endif
        TryStartProbe();
    }

    void Update()
    {
#if UNITY_ANDROID
        if (!recording &&
            Permission.HasUserAuthorizedPermission(Permission.Microphone) &&
            status.StartsWith("Requesting", StringComparison.OrdinalIgnoreCase))
        {
            TryStartProbe();
        }
#endif
    }

    void TryStartProbe()
    {
        status = "Starting FMOD probe...";
        UnityEngine.Debug.Log(LogPrefix + status);

        try
        {
            fmod = RuntimeManager.CoreSystem;
        }
        catch (Exception e)
        {
            status = "ERROR: RuntimeManager.CoreSystem failed";
            UnityEngine.Debug.LogError(LogPrefix + status + " :: " + e);
            return;
        }

        // Record driver bilgileri
        int numDrivers, numConnected;
        RESULT r = fmod.getRecordNumDrivers(out numDrivers, out numConnected);
        if (r != RESULT.OK)
        {
            status = "ERROR: getRecordNumDrivers failed: " + r;
            LogE(status);
            return;
        }

        Log($"Record devices: {numDrivers} (connected: {numConnected})");

        if (numDrivers <= 0)
        {
            status = "ERROR: No record drivers. (Android permission, audio route, or FMOD init timing)";
            LogE(status);
            return;
        }

        // Cihaz seçimi: keyword match, yoksa connected, yoksa 0
        int keywordMatch = -1;
        int firstConnected = -1;

        for (int i = 0; i < numDrivers; i++)
        {
            string name;
            System.Guid guidTmp;          // ✅ System.Guid (System.GUID değil)
            int sysRate;
            SPEAKERMODE spk;
            int ch;
            DRIVER_STATE state;

            r = fmod.getRecordDriverInfo(i, out name, 256, out guidTmp, out sysRate, out spk, out ch, out state);
            if (r != RESULT.OK)
            {
                UnityEngine.Debug.LogWarning(LogPrefix + $"getRecordDriverInfo failed for {i}: {r}");
                continue;
            }

            // Her cihaz satırında channel sayısı görünsün
            UnityEngine.Debug.Log(LogPrefix + $"Device[{i}] '{name}' sysRate={sysRate} ch={ch} mode={spk} state={state}");

            bool isConnected = (state & DRIVER_STATE.CONNECTED) != 0;
            if (firstConnected < 0 && isConnected) firstConnected = i;

            if (keywordMatch < 0 &&
                !string.IsNullOrEmpty(name) &&
                name.IndexOf(preferredDeviceKeyword, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                keywordMatch = i;
            }
        }

        if (keywordMatch >= 0) recDeviceId = keywordMatch;
        else if (preferConnectedDrivers && firstConnected >= 0) recDeviceId = firstConnected;
        else if (fallbackToFirstDevice) recDeviceId = 0;

        if (recDeviceId < 0)
        {
            status = "ERROR: Device not found. Change preferredDeviceKeyword.";
            LogE(status);
            return;
        }

        // Seçilen cihazın adını + kanal sayısını kesin olarak al
        {
            System.Guid guidSel;          // ✅ System.Guid
            RESULT rr = fmod.getRecordDriverInfo(
                recDeviceId,
                out recDeviceName, 256,
                out guidSel,
                out recDeviceSysRate,
                out recDeviceMode,
                out recDeviceChannels,
                out recDeviceState
            );

            if (rr != RESULT.OK)
            {
                recDeviceName = "unknown";
                recDeviceChannels = -1;
                recDeviceSysRate = sampleRate;
                recDeviceMode = SPEAKERMODE.DEFAULT;
                recDeviceState = 0;
                LogW($"getRecordDriverInfo(selected) failed: {rr}");
            }
        }

        Log($"Selected device. sysRate={recDeviceSysRate}, mode={recDeviceMode}, state={recDeviceState}");

        // Stereo mu?
        string stereoTag = (recDeviceChannels == 2) ? "STEREO ✅" :
                           (recDeviceChannels == 1) ? "MONO ❌" :
                           $"UNKNOWN(ch={recDeviceChannels})";
        Log($"Driver reports: {stereoTag}");

        // Bizim kayıt sound'umuzun kanal sayısı:
        // - Driver 2 veriyorsa 2 yakala
        // - 1 veriyorsa 1 yakala
        // - saçma/unknown ise 1
        actualChannels = (uint)((recDeviceChannels == 2) ? 2 : 1);

        // Ring buffer size (bytes)
        ringBytes = (uint)(ringSeconds * sampleRate * actualChannels * BytesPerSample);
        Log($"Capture format: actualChannels={actualChannels}, frameBytes={FrameBytes}, ringBytes={ringBytes}");

        // Create a user sound for recording
        CREATESOUNDEXINFO ex = new CREATESOUNDEXINFO();
        ex.cbsize = Marshal.SizeOf(typeof(CREATESOUNDEXINFO));
        ex.numchannels = (int)actualChannels;
        ex.defaultfrequency = sampleRate;
        ex.format = SOUND_FORMAT.PCM16;
        ex.length = ringBytes;

        r = fmod.createSound((string)null, MODE.OPENUSER | MODE.LOOP_NORMAL, ref ex, out recSound);
        if (r != RESULT.OK)
        {
            status = "ERROR: createSound failed: " + r;
            LogE(status);
            return;
        }

        // Start recording
        r = fmod.recordStart(recDeviceId, recSound, true);
        if (r != RESULT.OK)
        {
            status = "ERROR: recordStart failed: " + r;
            LogE(status);
            SafeReleaseSound();
            return;
        }

        recording = true;
        status = "Recording started. Probing...";
        Log(status);

        CancelInvoke();
        InvokeRepeating(nameof(Probe), 0.5f, probeIntervalSec);
    }

    void OnDisable()
    {
        CancelInvoke();

        if (recording && recDeviceId >= 0)
        {
            fmod.recordStop(recDeviceId);
            recording = false;
            Log("Recording stopped.");
        }

        SafeReleaseSound();
    }

    void SafeReleaseSound()
    {
        try
        {
            if (recSound.hasHandle())
            {
                recSound.release();
                Log("Sound released.");
            }
        }
        catch { /* ignore */ }
    }

    void Probe()
    {
        if (!recording) return;

        bool isRec;
        RESULT r = fmod.isRecording(recDeviceId, out isRec);
        if (r != RESULT.OK || !isRec) return;

        uint recordPos;
        r = fmod.getRecordPosition(recDeviceId, out recordPos);
        if (r != RESULT.OK) return;

        uint windowBytes = windowFrames * FrameBytes;

        uint writeBytes = recordPos * FrameBytes;
        uint startBytes = (writeBytes + ringBytes - windowBytes) % ringBytes;

        IntPtr p1, p2;
        uint len1, len2;

        r = recSound.@lock(startBytes, windowBytes, out p1, out p2, out len1, out len2);
        if (r != RESULT.OK) return;

        double sumL = 0.0, sumR = 0.0;
        int frames = 0;

        ProcessBlock(p1, len1, actualChannels, ref sumL, ref sumR, ref frames);
        if (len2 > 0) ProcessBlock(p2, len2, actualChannels, ref sumL, ref sumR, ref frames);

        recSound.unlock(p1, p2, len1, len2);

        if (frames <= 0) return;

        double rmsL = Math.Sqrt(sumL / frames);
        double rmsR = (actualChannels >= 2) ? Math.Sqrt(sumR / frames) : 0.0;

        lastRmsL = rmsL;
        lastRmsR = rmsR;

        string dominant;
        if (actualChannels < 2)
        {
            dominant = "MONO input";
        }
        else
        {
            dominant = "similar";
            if (rmsL > rmsR * 1.25) dominant = "LEFT dominant";
            else if (rmsR > rmsL * 1.25) dominant = "RIGHT dominant";
        }

        lastDominant = dominant;

        // Her log satırında name + ch prefix var ✅
        Log($"RMS_L={rmsL:F6} RMS_R={rmsR:F6} => {dominant}");
    }

    static void ProcessBlock(IntPtr ptr, uint byteLen, uint channels, ref double sumL, ref double sumR, ref int frames)
    {
        if (ptr == IntPtr.Zero || byteLen < 2) return;

        int n = (int)byteLen;
        byte[] buf = new byte[n];
        Marshal.Copy(ptr, buf, 0, n);

        if (channels < 2)
        {
            // Mono PCM16: [M_lo, M_hi] repeating
            for (int i = 0; i + 1 < n; i += 2)
            {
                short M = (short)(buf[i] | (buf[i + 1] << 8));
                double mf = M / 32768.0;
                sumL += mf * mf;
                frames++;
            }
            return;
        }

        // Stereo PCM16 little-endian: [L_lo, L_hi, R_lo, R_hi] repeating
        for (int i = 0; i + 3 < n; i += 4)
        {
            short L = (short)(buf[i] | (buf[i + 1] << 8));
            short R = (short)(buf[i + 2] | (buf[i + 3] << 8));

            double lf = L / 32768.0;
            double rf = R / 32768.0;

            sumL += lf * lf;
            sumR += rf * rf;
            frames++;
        }
    }

    void OnGUI()
    {
        if (!showOnScreen) return;

        GUIStyle s = new GUIStyle(GUI.skin.label);
        s.fontSize = 22;

        GUILayout.BeginArea(new Rect(20, 20, 1500, 260));
        GUILayout.Label("FMOD Stereo Probe", s);
        GUILayout.Label(status, s);

#if UNITY_ANDROID
        GUILayout.Label("Android mic permission: " +
                        (Permission.HasUserAuthorizedPermission(Permission.Microphone) ? "GRANTED" : "NOT GRANTED"), s);
#endif

        GUILayout.Label($"DeviceId={recDeviceId}  Name='{recDeviceName}'  DriverCh={recDeviceChannels}", s);
        GUILayout.Label($"SysRate={recDeviceSysRate}  Mode={recDeviceMode}  State={recDeviceState}", s);
        GUILayout.Label($"ActualCaptureCh={actualChannels}  WindowFrames={windowFrames}", s);

        if (recording)
        {
            GUILayout.Label($"RMS_L={lastRmsL:F6}  RMS_R={lastRmsR:F6}  => {lastDominant}", s);
        }

        GUILayout.EndArea();
    }
}