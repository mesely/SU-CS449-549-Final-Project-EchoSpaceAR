using UnityEngine;

public class YamnetIconPanMover2 : MonoBehaviour
{
    [Header("Target UI")]
    public RectTransform target; // YamnetIcon rect

    [Header("Movement")]
    [Tooltip("Max horizontal movement in pixels at pan=-1..+1.")]
    public float maxOffsetPx = 220f;

    [Tooltip("How fast the UI follows changes (bigger = faster).")]
    public float followSpeed = 10f;

    [Tooltip("Clamp incoming pan to [-1, 1].")]
    public bool clampPan = true;

    [Header("Optional - quiet/center behavior")]
    public bool recenterWhenLowConfidence = true;
    [Range(0f, 1f)]
    public float confidenceThreshold = 0.25f;

    private float _targetPan = 0f;
    private float _targetConf = 1f;
    private Vector2 _baseAnchoredPos;

    private void Awake()
    {
        if (target == null) target = GetComponent<RectTransform>();
        _baseAnchoredPos = target.anchoredPosition;
    }

    public void SetPan(float pan, float confidence = 1f)
    {
        if (clampPan) pan = Mathf.Clamp(pan, -1f, 1f);
        _targetPan = pan;
        _targetConf = Mathf.Clamp01(confidence);
    }

    private void Update()
    {
        if (target == null) return;

        float pan = _targetPan;

        if (recenterWhenLowConfidence && _targetConf < confidenceThreshold)
            pan = 0f;

        float desiredX = _baseAnchoredPos.x + pan * maxOffsetPx;

        Vector2 cur = target.anchoredPosition;
        Vector2 desired = new Vector2(desiredX, _baseAnchoredPos.y);

        target.anchoredPosition = Vector2.Lerp(
            cur, desired, 1f - Mathf.Exp(-followSpeed * Time.deltaTime)
        );
    }
}
