using UnityEngine;
using UnityEngine.Android;
using UnityEngine.UI;

public class MicDebug : MonoBehaviour
{
    [Header("UI (Optional)")]
    public Text uiText;                 // Legacy Text
    public TMPro.TMP_Text tmpText;      // TMP (varsa)

    [Header("Mic Settings")]
    public int sampleRate = 48000;
    public int clipLengthSec = 1;

    private string deviceName = null;   // null = DEFAULT mic (Quest)
    private AudioClip clip;

    void Start()
    {
        // Request mic permission
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
            Permission.RequestUserPermission(Permission.Microphone);

        Invoke(nameof(StartMic), 0.5f); // izin ekranı açılırsa biraz beklesin
    }

    void StartMic()
    {
        bool perm = Permission.HasUserAuthorizedPermission(Permission.Microphone);

        string devices = (Microphone.devices.Length == 0)
            ? "(none)"
            : string.Join(" | ", Microphone.devices);

        LogUI($"perm={perm}\ndevCount={Microphone.devices.Length}\ndevices={devices}\nusing=DEFAULT(None)");

        if (!perm) return;

        // DEFAULT mic ile başlat (Quest mic)
        clip = Microphone.Start(deviceName, true, clipLengthSec, sampleRate);

        if (clip == null)
            LogUI("Microphone.Start failed (clip null)");
    }

    void Update()
    {
        if (clip == null) return;
        if (!Microphone.IsRecording(deviceName)) return;

        int pos = Microphone.GetPosition(deviceName);
        if (pos < 256) return;

        float[] data = new float[256];
        clip.GetData(data, Mathf.Max(0, pos - data.Length));

        float sum = 0f;
        for (int i = 0; i < data.Length; i++) sum += data[i] * data[i];
        float rms = Mathf.Sqrt(sum / data.Length);

        string devices = (Microphone.devices.Length == 0)
            ? "(none)"
            : string.Join(" | ", Microphone.devices);

        LogUI(
            $"perm={Permission.HasUserAuthorizedPermission(Permission.Microphone)}\n" +
            $"devCount={Microphone.devices.Length}\n" +
            $"devices={devices}\n" +
            $"using=DEFAULT(None)\n" +
            $"pos={pos}  rms={rms:0.0000}"
        );
    }

    void LogUI(string s)
    {
        Debug.Log("[MicDebug] " + s);
        if (uiText) uiText.text = s;
        if (tmpText) tmpText.text = s;
    }
}
