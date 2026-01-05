using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class BackendHttpClient : MonoBehaviour
{
    [Header("Backend URLs (example)")]
    public string baseUrl = "http://192.168.1.36:8000"; // your Mac/server
    public string clientHelloPath = "/client_hello";
    public string audioChunkPath = "/audio_chunk";
    public string eventsPath = "/events";

    [Header("Session")]
    public string sessionId;
    public float pollIntervalSeconds = 0.5f;

    private double _lastEventsTimestampUnix = 0.0;

    public event Action<BackendEvent> OnEventReceived;

    private void Awake()
    {
        //if (string.IsNullOrEmpty(sessionId))
        //{
        //    sessionId = Guid.NewGuid().ToString();
        //}
        sessionId = "default";
    }

    private void Start()
    {
        StartCoroutine(SendClientHello());
        StartCoroutine(PollEventsLoop());
    }

    private double GetUnixTime()
    {
        return (DateTime.UtcNow -
                new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc))
            .TotalSeconds;
    }

    private IEnumerator SendClientHello()
    {
        var reqBody = new ClientHelloRequest
        {
            session_id = sessionId,
            timestamp_unix = GetUnixTime(),
            device_id = SystemInfo.deviceUniqueIdentifier,
            app_version = Application.version,
            platform = Application.platform.ToString(),
            has_ar_kit = (Application.platform == RuntimePlatform.IPhonePlayer),
            has_ar_core = (Application.platform == RuntimePlatform.Android),
            preferred_samplerate_hz = 16000,
            channels = 1,
            sample_format = "float32"
        };

        string json = JsonUtility.ToJson(reqBody);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest www = new UnityWebRequest(baseUrl + clientHelloPath, "POST"))
        {
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("[BackendHttpClient] client_hello failed: " + www.error);
            }
            else
            {
                Debug.Log("[BackendHttpClient] client_hello OK, response: " + www.downloadHandler.text);
            }
        }
    }

    private IEnumerator PollEventsLoop()
    {
        while (true)
        {
            yield return PollEventsOnce();
            yield return new WaitForSeconds(pollIntervalSeconds);
        }
    }

    private IEnumerator PollEventsOnce()
    {
        string url = $"{baseUrl}{eventsPath}?session_id={sessionId}&since_unix={_lastEventsTimestampUnix}";

        using (UnityWebRequest www = UnityWebRequest.Get(url))
        {
            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning("[BackendHttpClient] Poll events failed: " + www.error);
                yield break;
            }

            string json = www.downloadHandler.text;
            if (string.IsNullOrEmpty(json))
            {
                yield break;
            }

            try
            {
                var resp = JsonUtility.FromJson<BackendEventsResponse>(json);
                if (resp != null && resp.events != null)
                {
                    foreach (var ev in resp.events)
                    {
                        OnEventReceived?.Invoke(ev);
                        Debug.Log($"[BackendHttpClient] Event kind={ev.kind}, time={ev.timestamp_unix}");
                    }
                    _lastEventsTimestampUnix = resp.last_timestamp_unix;
                }
            }
            catch (Exception ex)
            {
                Debug.LogError("[BackendHttpClient] Failed to parse events: " + ex);
            }
        }
    }

    // -------- AUDIO CHUNK SENDING --------

    public IEnumerator SendAudioChunk(AudioChunkRequest chunk)
    {
        string json = JsonUtility.ToJson(chunk);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

        string url = baseUrl + audioChunkPath;

        using (UnityWebRequest www = new UnityWebRequest(url, "POST"))
        {
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning("[BackendHttpClient] SendAudioChunk failed: " + www.error);
            }
            else
            {
                // For debugging you can log a tiny snippet:
                 Debug.Log("[BackendHttpClient] AudioChunk sent, response: " + www.downloadHandler.text);
            }
        }
    }
}
