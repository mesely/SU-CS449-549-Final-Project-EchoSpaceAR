using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class IconManager : MonoBehaviour
{
    [Header("UI References")]
    public Image iconImage;
    public TextMeshProUGUI labelText;  // optional, can be null

    [Header("Label → Sprite mapping")]
    public List<LabelSpritePair> labelSprites;

    [Header("Fallback")]
    public Sprite defaultSprite;
    public string lastLabel = "";

    private Dictionary<string, Sprite> _map;

    [Serializable]
    public class LabelSpritePair
    {
        public string label;    // e.g. "speech", "siren_emergency"
        public Sprite sprite;
    }

    private void Awake()
    {
        _map = new Dictionary<string, Sprite>(StringComparer.OrdinalIgnoreCase);
        foreach (var pair in labelSprites)
        {
            if (!string.IsNullOrEmpty(pair.label) && pair.sprite != null && !_map.ContainsKey(pair.label))
            {
                _map.Add(pair.label, pair.sprite);
            }
        }
    }

    /// <summary>
    /// Call this when you get a new YAMNet dominant label.
    /// </summary>
    public void SetActiveLabel(string label)
    {
        if (string.IsNullOrEmpty(label)) return;

        lastLabel = label;

        // Find sprite
        Sprite sprite;
        if (!_map.TryGetValue(label, out sprite))
        {
            sprite = defaultSprite;
        }

        // Update icon
        if (iconImage != null)
        {
            iconImage.sprite = sprite;
            iconImage.enabled = (sprite != null);
        }

        // Update text (optional)
        if (labelText != null)
        {
            labelText.text = label.Replace("_", " ");
        }
    }
}
