using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class IconView : MonoBehaviour
{
    [Header("UI References")]
    public Image backgroundImage;
    public Image iconImage;
    public TextMeshProUGUI labelText;

    [Header("State")]
    public string label;      // fine-grain label
    public string mappedType; // "speech", "alarm", ...
    public string priority;   // "low", "medium", "high"

    private float _timeCreated;
    private float _duration = 3f;
    private bool _autoExpire = true;

    private void Awake()
    {
        _timeCreated = Time.time;
    }

    private void Update()
    {
        if (!_autoExpire) return;

        float age = Time.time - _timeCreated;
        if (age > _duration)
        {
            Destroy(gameObject);
        }
    }

    public void Initialize(
        string label,
        string mappedType,
        string priority,
        Sprite sprite,
        Color baseColor,
        string displayText,
        float duration = 3f,
        bool autoExpire = true)
    {
        this.label = label;
        this.mappedType = mappedType;
        this.priority = priority;

        _timeCreated = Time.time;
        _duration = duration;
        _autoExpire = autoExpire;

        if (iconImage != null)
        {
            iconImage.sprite = sprite;
            iconImage.enabled = sprite != null;
        }

        if (backgroundImage != null)
        {
            backgroundImage.color = ApplyPriorityTint(baseColor, priority);
        }

        if (labelText != null)
        {
            labelText.text = displayText;
        }
    }

    private Color ApplyPriorityTint(Color baseColor, string priority)
    {
        switch (priority)
        {
            case "high":
                return Color.Lerp(baseColor, Color.red, 0.7f);
            case "medium":
                return Color.Lerp(baseColor, new Color(1f, 0.8f, 0f), 0.7f);
            case "low":
            default:
                return baseColor;
        }
    }
}
