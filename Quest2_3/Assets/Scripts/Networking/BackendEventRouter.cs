using System;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class BackendEventRouter : MonoBehaviour
{
    [Header("References")]
    public BackendHttpClient backend;
    public TMP_Text statusText;
    public TMP_Text subtitleText;
    public TMP_Text summaryText;

    // Keep last values, useful if you later add history UI
    private string _lastYamnetLabel = "";
    private double _lastYamnetProb = 0.0;
    private string _lastSubtitle = "";
    private string _lastSummary = "";

    public IconManager yamnetIconDisplay;


    private void OnEnable()
    {
        if (backend != null)
        {
            backend.OnEventReceived += HandleBackendEvent;
        }
    }

    private void OnDisable()
    {
        if (backend != null)
        {
            backend.OnEventReceived -= HandleBackendEvent;
        }
    }

    private void Start()
    {
        if (statusText != null)
        {
            statusText.text = "Status: Connecting / waiting for events...";
        }
        if (subtitleText != null)
        {
            subtitleText.text = "";
        }
        if (summaryText != null)
        {
            summaryText.text = "";
        }
    }

    private void HandleBackendEvent(BackendEvent ev)
    {
        // Update status text with latest event kind & time
        if (statusText != null)
        {
            statusText.text = $"Status: Last event kind={ev.kind}, t={ev.timestamp_unix:F1}";
        }

        switch (ev.kind)
        {
            case "yamnet":
                HandleYamnet(ev.yamnet);
                break;

            case "stt":
                HandleStt(ev.stt);
                break;

            case "llm":
                HandleLlm(ev.llm);
                break;

            case "important":
                HandleImportant(ev.important);
                break;

            default:
                Debug.Log($"[BackendEventRouter] Unknown event kind: {ev.kind}");
                break;
        }
    }

    private void HandleYamnet(YamnetEvent y)
    {
        if (y == null)
            return;
        if (y.dominant_label == "other")
        {
            _lastYamnetLabel = y.dominant_label;
            _lastYamnetProb = y.dominant_prob;
        }
        else
        {
            _lastYamnetLabel = y.top5[2].label;
            _lastYamnetProb = y.top5[2].prob;
        }
        _lastYamnetLabel = y.dominant_label;
        _lastYamnetProb = y.dominant_prob;

        string topStr = "";
        if (y.top5 != null && y.top5.Length > 0)
        {
            // e.g., "speech(0.95), dog(0.03), other(0.02)"
            int count = Mathf.Min(3, y.top5.Length);
            for (int i = 0; i < count; i++)
            {
                var lp = y.top5[i];
                topStr += $"{lp.label}({lp.prob:F2})";
                if (i < count - 1) topStr += ", ";
            }
        }

        Debug.Log($"[BackendEventRouter] YAMNet dominant={_lastYamnetLabel} prob={_lastYamnetProb:F2}");

        // Optional: show current sound state in status or somewhere else
        if (statusText != null)
        {
            statusText.text = $"Sound: {_lastYamnetLabel} ({_lastYamnetProb:F2}), SPL={y.spl_dbfs:F1} dBFS";
            yamnetIconDisplay.SetActiveLabel(_lastYamnetLabel);
        }
    }

    private void HandleStt(SttSegment s)
    {
        if (s == null || string.IsNullOrEmpty(s.text))
            return;

        _lastSubtitle = s.text;
        Debug.Log($"[BackendEventRouter] STT: {s.text}");

        if (subtitleText != null)
        {
            subtitleText.text = _lastSubtitle;
        }
    }

    private void HandleLlm(LlmSummary l)
    {
        if (l == null)
            return;

        _lastSummary = l.user_message;
        Debug.Log($"[BackendEventRouter] LLM user_message: {l.user_message}");

        if (summaryText != null)
        {
            summaryText.text = _lastSummary;
        }
    }

    private void HandleImportant(ImportantEvent imp)
    {
        if (imp == null)
            return;

        // For now, just log and maybe prepend to status line
        string msg = $"IMPORTANT [{imp.priority}] {imp.mapped_type}: {imp.description}";
        Debug.LogWarning("[BackendEventRouter] " + msg);

        if (statusText != null)
        {
            statusText.text = "!!! " + msg;
        }

        // Later, this will drive red icons / flashes in AR HUD.
    }
}
