using UnityEngine;

public class FollowCameraFixedDistance : MonoBehaviour
{
    public Transform cam;
    public float distance = 2f;

    void LateUpdate()
    {
        if (!cam) return;
        transform.position = cam.position + cam.forward * distance;
        transform.rotation = Quaternion.identity;
    }
}
