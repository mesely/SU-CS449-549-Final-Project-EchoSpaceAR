using System;
using System.Collections;
using System.Runtime.InteropServices;
using UnityEngine;

using FMOD;
using FMODUnity;

#if UNITY_ANDROID
using UnityEngine.Android;
#endif

[RequireComponent(typeof(BackendHttpClient))]
public class AudioCaptureController : MonoBehaviour
{
    [Header("UI Direction Output")]
    public YamnetIconPanMover2 panMover;

    [Header("Backend Audio Format")]
    public int targetSampleRate = 16000;
    public float chunkDurationSeconds = 0.5f;
    public int ringSeconds = 10;

    [Header("Device Selection")]
    public string preferredDeviceKeyword = "Wireless"; // Rode Wireless GO II
    public bool preferConnectedDrivers = true;
    public bool fallbackToFirstDevice = true;

    [Header("Capture Mode")]
    [Tooltip("0=use driver channels, 1=mono, 2=stereo (still downmix to mono before sending)")]
    public int forceCaptureChannels = 0;

    [Header("Left/Right Detection")]
    [Tooltip("Ignore direction when too quiet (RMS gate).")]
    public float minRmsGate = 0.002f;

    [Tooltip("Enter Left/Right when |dB| exceeds this.")]
    public float enterThresholdDb = 3.0f;

    [Tooltip("Return to Center when |dB| drops below this (hysteresis).")]
    public float exitThresholdDb = 1.5f;

    [Range(0.01f, 1f)]
    public float smoothingAlpha = 0.25f;

    [Tooltip("Clamp dB diff to this for pan mapping.")]
    public float panClampDb = 12f;

    [Header("UnityEngine.Debug UI")]
    public bool logEachChunk = true;
    public bool showOnScreen = true;

    // Backend
    private BackendHttpClient _backend;
    private int _seq = 0;

    // FMOD
    private FMOD.System _fmod;
    private Sound _recSound;
    private int _recDeviceId = -1;
    private bool _recording = false;

    private string _recDeviceName = "n/a";
    private int _driverChannels = -1;
    private int _driverSysRate = -1;
    private SPEAKERMODE _driverMode = SPEAKERMODE.DEFAULT;
    private DRIVER_STATE _driverState = 0;

    // Capture format (PCM16 ring)
    private int _captureSampleRate = 48000;
    private uint _captureChannels = 1;
    private const uint BytesPerSample = 2; // PCM16
    private uint FrameBytes => _captureChannels * BytesPerSample;

    private uint _ringBytes;
    private uint _chunkFramesCapture;
    private uint _chunkBytesCapture;

    // Buffers
    private byte[] _pcm16Bytes;        // interleaved capture bytes for chunk
    private float[] _monoCapture;      // mono float @ capture rate (downmix)
    private float[] _outFloat;         // mono float @ targetSampleRate (sent)
    private float[] _tmpResample;      // scratch

    // Direction state
    private double _dbDiffSmooth = 0.0;     // + left louder, - right louder
    private string _lrState = "Center";     // Left/Right/Center/Mono/Quiet
    private float _pan = 0f;               // -1..+1
    private float _confidence = 0f;        // 0..1
    private float _lastRmsL = 0f, _lastRmsR = 0f;

    private string LogPrefix => $"[FMODCapture][dev={_recDeviceId} name='{_recDeviceName}' ch={_driverChannels}] ";

    private void Awake()
    {
        _backend = GetComponent<BackendHttpClient>();
    }

    private void Start()
    {
        Application.runInBackground = true;

#if UNITY_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            UnityEngine.Debug.Log(LogPrefix + "Requesting MICROPHONE permission...");
            Permission.RequestUserPermission(Permission.Microphone);
            return;
        }
#endif
        StartFmodRecording();
        StartCoroutine(CaptureLoop());
    }

    private void OnDisable()
    {
        StopFmodRecording();
    }

    // -------------------------
    // FMOD init + device select
    // -------------------------

    private void StartFmodRecording()
    {
        try
        {
            _fmod = RuntimeManager.CoreSystem;
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError(LogPrefix + "ERROR: RuntimeManager.CoreSystem failed :: " + e);
            return;
        }

        int numDrivers, numConnected;
        var r = _fmod.getRecordNumDrivers(out numDrivers, out numConnected);
        if (r != RESULT.OK || numDrivers <= 0)
        {
            UnityEngine.Debug.LogError(LogPrefix + $"ERROR: getRecordNumDrivers failed or no devices. r={r}, n={numDrivers}");
            return;
        }

        // Pick best device by score: keyword + stereo + connected
        int bestDevice = -1;
        int bestScore = int.MinValue;

        for (int i = 0; i < numDrivers; i++)
        {
            string name;
            Guid guidTmp;
            int sysRate;
            SPEAKERMODE spk;
            int ch;
            DRIVER_STATE state;

            r = _fmod.getRecordDriverInfo(i, out name, 256, out guidTmp, out sysRate, out spk, out ch, out state);
            if (r != RESULT.OK) continue;

            bool isConnected = (state & DRIVER_STATE.CONNECTED) != 0;
            bool keywordHit =
                !string.IsNullOrEmpty(preferredDeviceKeyword) &&
                !string.IsNullOrEmpty(name) &&
                name.IndexOf(preferredDeviceKeyword, StringComparison.OrdinalIgnoreCase) >= 0;

            int score = 0;
            if (keywordHit) score += 100;
            if (ch >= 2) score += 50;
            if (preferConnectedDrivers && isConnected) score += 10;
            if (sysRate >= 48000) score += 5;

            UnityEngine.Debug.Log(LogPrefix + $"Device[{i}] '{name}' sysRate={sysRate} ch={ch} connected={isConnected} score={score}");

            if (score > bestScore)
            {
                bestScore = score;
                bestDevice = i;
            }
        }

        if (bestDevice < 0)
        {
            if (fallbackToFirstDevice) bestDevice = 0;
            else
            {
                UnityEngine.Debug.LogError(LogPrefix + "ERROR: No suitable record device found.");
                return;
            }
        }

        _recDeviceId = bestDevice;

        // Read selected driver info (cache)
        {
            Guid guidSel;
            r = _fmod.getRecordDriverInfo(_recDeviceId, out _recDeviceName, 256, out guidSel,
                out _driverSysRate, out _driverMode, out _driverChannels, out _driverState);

            if (r != RESULT.OK)
            {
                UnityEngine.Debug.LogWarning(LogPrefix + $"getRecordDriverInfo(selected) failed: {r}");
                _recDeviceName = "unknown";
                _driverSysRate = targetSampleRate;
                _driverChannels = 1;
            }
        }

        // Decide capture channels
        if (forceCaptureChannels == 1) _captureChannels = 1;
        else if (forceCaptureChannels == 2) _captureChannels = 2;
        else _captureChannels = (uint)((_driverChannels >= 2) ? 2 : 1);

        // Record at driver rate for stability; resample to target
        _captureSampleRate = (_driverSysRate > 0) ? _driverSysRate : 48000;

        // sizes
        _ringBytes = (uint)(ringSeconds * _captureSampleRate * _captureChannels * BytesPerSample);

        _chunkFramesCapture = (uint)Mathf.RoundToInt(_captureSampleRate * chunkDurationSeconds);
        _chunkBytesCapture = _chunkFramesCapture * FrameBytes;

        UnityEngine.Debug.Log(LogPrefix + $"Selected '{_recDeviceName}' driverRate={_driverSysRate} driverCh={_driverChannels}");
        UnityEngine.Debug.Log(LogPrefix + $"Capture rate={_captureSampleRate}, capCh={_captureChannels}, chunkFrames={_chunkFramesCapture}, ringBytes={_ringBytes}");

        // Create record sound (PCM16 ring)
        CREATESOUNDEXINFO ex = new CREATESOUNDEXINFO();
        ex.cbsize = Marshal.SizeOf(typeof(CREATESOUNDEXINFO));
        ex.numchannels = (int)_captureChannels;
        ex.defaultfrequency = _captureSampleRate;
        ex.format = SOUND_FORMAT.PCM16;
        ex.length = _ringBytes;

        r = _fmod.createSound((string)null, MODE.OPENUSER | MODE.LOOP_NORMAL, ref ex, out _recSound);
        if (r != RESULT.OK)
        {
            UnityEngine.Debug.LogError(LogPrefix + "ERROR: createSound failed: " + r);
            return;
        }

        r = _fmod.recordStart(_recDeviceId, _recSound, true);
        if (r != RESULT.OK)
        {
            UnityEngine.Debug.LogError(LogPrefix + "ERROR: recordStart failed: " + r);
            SafeReleaseSound();
            return;
        }

        _recording = true;

        // Buffers
        _pcm16Bytes = new byte[_chunkBytesCapture];
        _monoCapture = new float[_chunkFramesCapture];

        int outFrames = Mathf.RoundToInt(targetSampleRate * chunkDurationSeconds);
        _outFloat = new float[outFrames];
        _tmpResample = new float[outFrames];

        UnityEngine.Debug.Log(LogPrefix + "Recording started.");
    }

    private void StopFmodRecording()
    {
        try
        {
            if (_recording && _recDeviceId >= 0)
            {
                _fmod.recordStop(_recDeviceId);
                _recording = false;
            }
        }
        catch { }
        SafeReleaseSound();
    }

    private void SafeReleaseSound()
    {
        try
        {
            if (_recSound.hasHandle())
                _recSound.release();
        }
        catch { }
    }

    // -------------------------
    // Loop
    // -------------------------

    private IEnumerator CaptureLoop()
    {
        yield return new WaitForSeconds(0.5f);

        while (true)
        {
            yield return new WaitForSeconds(chunkDurationSeconds);
            if (_recording) CaptureAndSend();
        }
    }

    private void CaptureAndSend()
    {
        // record position
        bool isRec;
        var r = _fmod.isRecording(_recDeviceId, out isRec);
        if (r != RESULT.OK || !isRec) return;

        uint recordPosFrames;
        r = _fmod.getRecordPosition(_recDeviceId, out recordPosFrames);
        if (r != RESULT.OK) return;

        uint writeBytes = recordPosFrames * FrameBytes;
        uint startBytes = (writeBytes + _ringBytes - _chunkBytesCapture) % _ringBytes;

        IntPtr p1, p2;
        uint len1, len2;
        r = _recSound.@lock(startBytes, _chunkBytesCapture, out p1, out p2, out len1, out len2);
        if (r != RESULT.OK) return;

        int copied = 0;
        if (len1 > 0 && p1 != IntPtr.Zero)
        {
            Marshal.Copy(p1, _pcm16Bytes, copied, (int)len1);
            copied += (int)len1;
        }
        if (len2 > 0 && p2 != IntPtr.Zero)
        {
            Marshal.Copy(p2, _pcm16Bytes, copied, (int)len2);
            copied += (int)len2;
        }

        _recSound.unlock(p1, p2, len1, len2);

        if (copied < _pcm16Bytes.Length) return;

        // Direction estimate (if stereo)
        EstimateLeftRight(_pcm16Bytes, _captureChannels, (int)_chunkFramesCapture);

        if (panMover != null)
        {
            panMover.SetPan(_pan, _confidence);
        }


        // Convert to mono float @ capture rate (downmix if stereo)
        Pcm16InterleavedToMonoFloat(_pcm16Bytes, _captureChannels, _monoCapture);

        // Resample to target rate
        if (_captureSampleRate == targetSampleRate)
        {
            int n = Mathf.Min(_monoCapture.Length, _outFloat.Length);
            Array.Copy(_monoCapture, 0, _outFloat, 0, n);
            if (n < _outFloat.Length) Array.Clear(_outFloat, n, _outFloat.Length - n);
        }
        else
        {
            ResampleLinear(_monoCapture, _captureSampleRate, _outFloat, targetSampleRate);
        }

        if (logEachChunk)
        {
            UnityEngine.Debug.Log($"[Dir] {_lrState} pan={_pan:F2} conf={_confidence:F2} | rmsL={_lastRmsL:F4} rmsR={_lastRmsR:F4}");
        }

        // Send to backend (same schema as your old script)
        byte[] pcmBytes = FloatArrayToByteArray(_outFloat);
        string pcmBase64 = Convert.ToBase64String(pcmBytes);

        double nowUnix = GetUnixTime();
        double startUnix = nowUnix - chunkDurationSeconds;

        var chunkReq = new AudioChunkRequest
        {
            session_id = _backend.sessionId,
            timestamp_unix = nowUnix,
            seq = _seq++,
            samplerate_hz = targetSampleRate,
            channels = 1,
            sample_format = "float32",
            frame_count = _outFloat.Length,
            device_unix_time_start = startUnix,
            device_unix_time_end = nowUnix,
            pcm_base64 = pcmBase64,

            // OPTIONAL: If your C# AudioChunkRequest supports extra fields, add them:
            // pan = _pan,
            // pan_confidence = _confidence,
            // pan_state = _lrState
        };

        StartCoroutine(_backend.SendAudioChunk(chunkReq));
    }

    // -------------------------
    // Left/right estimation
    // -------------------------

    private void EstimateLeftRight(byte[] pcm16, uint channels, int frames)
    {
        if (channels < 2)
        {
            _lrState = "Mono";
            _pan = 0f;
            _confidence = 0f;
            _lastRmsL = 0f;
            _lastRmsR = 0f;
            return;
        }

        // Compute RMS L/R
        double sumL = 0.0, sumR = 0.0;
        int idx = 0;

        for (int i = 0; i < frames; i++)
        {
            short L = (short)(pcm16[idx] | (pcm16[idx + 1] << 8));
            short R = (short)(pcm16[idx + 2] | (pcm16[idx + 3] << 8));

            double lf = L / 32768.0;
            double rf = R / 32768.0;

            sumL += lf * lf;
            sumR += rf * rf;

            idx += 4;
        }

        double rmsL = Math.Sqrt(sumL / frames);
        double rmsR = Math.Sqrt(sumR / frames);

        _lastRmsL = (float)rmsL;
        _lastRmsR = (float)rmsR;

        // Gate quiet
        if (rmsL < minRmsGate && rmsR < minRmsGate)
        {
            _lrState = "Quiet";
            _pan = 0f;
            _confidence = 0f;
            return;
        }

        // dB difference: + left louder, - right louder
        double ratio = (rmsR > 1e-9) ? (rmsL / rmsR) : 1e9;
        double dbDiff = 20.0 * Math.Log10(ratio);

        // Smooth
        _dbDiffSmooth = (1.0 - smoothingAlpha) * _dbDiffSmooth + smoothingAlpha * dbDiff;

        // Pan mapping
        double panRaw = _dbDiffSmooth / panClampDb;
        if (panRaw > 1.0) panRaw = 1.0;
        if (panRaw < -1.0) panRaw = -1.0;
        _pan = (float)panRaw;

        // Confidence from |db|
        double absDb = Math.Abs(_dbDiffSmooth);
        _confidence = (float)Mathf.Clamp01((float)(absDb / enterThresholdDb));

        // Hysteresis
        switch (_lrState)
        {
            case "Left":
                if (_dbDiffSmooth < exitThresholdDb) _lrState = "Center";
                break;
            case "Right":
                if (_dbDiffSmooth > -exitThresholdDb) _lrState = "Center";
                break;
            default:
                if (_dbDiffSmooth > enterThresholdDb) _lrState = "Left";
                else if (_dbDiffSmooth < -enterThresholdDb) _lrState = "Right";
                else _lrState = "Center";
                break;
        }
    }

    // -------------------------
    // Audio conversion helpers
    // -------------------------

    private static void Pcm16InterleavedToMonoFloat(byte[] pcm16, uint channels, float[] monoOut)
    {
        int frameCount = monoOut.Length;

        if (channels < 2)
        {
            int bi = 0;
            for (int i = 0; i < frameCount; i++)
            {
                short s = (short)(pcm16[bi] | (pcm16[bi + 1] << 8));
                monoOut[i] = s / 32768f;
                bi += 2;
            }
            return;
        }

        int idx = 0;
        for (int i = 0; i < frameCount; i++)
        {
            short L = (short)(pcm16[idx] | (pcm16[idx + 1] << 8));
            short R = (short)(pcm16[idx + 2] | (pcm16[idx + 3] << 8));
            monoOut[i] = 0.5f * ((L / 32768f) + (R / 32768f));
            idx += 4;
        }
    }

    private static void ResampleLinear(float[] inBuf, int inRate, float[] outBuf, int outRate)
    {
        int inLen = inBuf.Length;
        int outLen = outBuf.Length;

        if (inLen <= 1)
        {
            Array.Clear(outBuf, 0, outLen);
            return;
        }

        double ratio = (double)inRate / outRate;

        for (int i = 0; i < outLen; i++)
        {
            double srcPos = i * ratio;
            int i0 = (int)Math.Floor(srcPos);
            int i1 = Math.Min(i0 + 1, inLen - 1);
            double frac = srcPos - i0;

            float v0 = inBuf[i0];
            float v1 = inBuf[i1];
            outBuf[i] = (float)(v0 + (v1 - v0) * frac);
        }
    }

    private static byte[] FloatArrayToByteArray(float[] data)
    {
        int len = data.Length;
        byte[] bytes = new byte[len * 4];

        for (int i = 0; i < len; i++)
        {
            byte[] fBytes = BitConverter.GetBytes(data[i]);
            Array.Copy(fBytes, 0, bytes, i * 4, 4);
        }
        return bytes;
    }

    private static double GetUnixTime()
    {
        return (DateTime.UtcNow -
                new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc))
            .TotalSeconds;
    }

    private void OnGUI()
    {
        /*
        if (!showOnScreen) return;

        GUIStyle s = new GUIStyle(GUI.skin.label) { fontSize = 20 };

        GUILayout.BeginArea(new Rect(20, 20, 1600, 320));
        GUILayout.Label("FMOD Capture + L/R Direction", s);
        GUILayout.Label($"Device: {_recDeviceId} '{_recDeviceName}' driverCh={_driverChannels} driverRate={_driverSysRate}", s);
        GUILayout.Label($"Capture: rate={_captureSampleRate} ch={_captureChannels} | Backend: rate={targetSampleRate} mono", s);
        GUILayout.Label($"Dir: {_lrState} pan={_pan:F2} conf={_confidence:F2}", s);
        GUILayout.Label($"RMS: L={_lastRmsL:F4} R={_lastRmsR:F4} (gate={minRmsGate})", s);
        GUILayout.EndArea();
        */
    }
}
