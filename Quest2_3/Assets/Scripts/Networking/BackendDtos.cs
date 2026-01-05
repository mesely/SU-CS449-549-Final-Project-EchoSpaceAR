using System;
using UnityEngine;

[Serializable]
public class ClientHelloRequest
{
    public string session_id;
    public double timestamp_unix;
    public string device_id;
    public string app_version;
    public string platform;
    public bool has_ar_kit;
    public bool has_ar_core;
    public int preferred_samplerate_hz;
    public int channels;
    public string sample_format;
}

[Serializable]
public class AudioChunkRequest
{
    public string session_id;
    public double timestamp_unix;
    public int seq;
    public int samplerate_hz;
    public int channels;
    public string sample_format;
    public int frame_count;
    public double device_unix_time_start;
    public double device_unix_time_end;
    public string pcm_base64;
}

// ----- Events response -----

[Serializable]
public class BackendEventsResponse
{
    public BackendEvent[] events;
    public double last_timestamp_unix;
}

[Serializable]
public class BackendEvent
{
    public string kind;           // "yamnet" | "stt" | "llm" | "important"
    public double timestamp_unix;
    public string session_id;

    public YamnetEvent yamnet;    // filled if kind == "yamnet"
    public SttSegment stt;        // filled if kind == "stt"
    public LlmSummary llm;        // filled if kind == "llm"
    public ImportantEvent important; // filled if kind == "important"
}

[Serializable]
public class YamnetEvent
{
    public double window_start_unix;
    public double window_end_unix;
    public YamnetLabelProb[] top5;
    public string dominant_label;
    public double dominant_prob;
    public double spl_dbfs;
}

[Serializable]
public class YamnetLabelProb
{
    public string label;
    public double prob;
}

[Serializable]
public class SttSegment
{
    public string segment_id;
    public double start_unix;
    public double end_unix;
    public double eff_window_s;
    public int samplerate_hz;
    public string text;
    public string language;
    public double confidence;
}

[Serializable]
public class LlmSummary
{
    public double window_start_unix;
    public double window_end_unix;
    public string brief_summary;
    public string user_message;
    public LlmImportantEvent[] important_events;
}

[Serializable]
public class LlmImportantEvent
{
    public string type;        // "speech" | "alarm" | ...
    public string description;
    public string priority;    // "low" | "medium" | "high"
}

[Serializable]
public class ImportantEvent
{
    public string event_id;
    public double event_time_unix;
    public string source;      // "yamnet" | "llm" | "stt"
    public string label;       // reduced label
    public string mapped_type; // "speech" | "alarm" | ...
    public string priority;    // "low" | "medium" | "high"
    public double confidence;
    public string description;
}
