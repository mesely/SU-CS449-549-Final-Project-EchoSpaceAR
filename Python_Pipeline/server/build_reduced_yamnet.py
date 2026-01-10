# build_reduced_yamnet.py
import os, csv, json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ---- EDIT: your reduced mapping (example: "safety_12") ----
REDUCED_MAP = {
  "speech": ["Speech","Child speech, kid speaking","Conversation","Narration, monologue","Whispering"],
  "crowd": ["Chatter","Crowd","Hubbub, speech noise, speech babble","Cheering","Applause","Children playing"],
  "music": ["Music","Musical instrument","Song","Background music","Theme music","Soundtrack music","Jingle (music)","Vocal music","A capella"],
  "dog": ["Dog","Bark","Yip","Howl","Bow-wow","Growling","Whimper (dog)"],
  "cat": ["Cat","Purr","Meow","Hiss","Caterwaul"],
  "bird": ["Bird","Bird vocalization, bird call, bird song","Chirp, tweet","Squawk","Pigeon, dove","Coo","Owl","Hoot","Bird flight, flapping wings","Crow","Caw"],
  "vehicle_horn": ["Vehicle horn, car horn, honking","Toot","Air horn, truck horn","Foghorn"],
  "traffic_road": ["Motor vehicle (road)","Traffic noise, roadway noise","Car passing by","Race car, auto racing","Skidding","Tire squeal"],
  "car_bus_truck": ["Car","Bus","Truck","Ice cream truck, ice cream van"],
  "sirens": ["Siren","Civil defense siren","Emergency vehicle","Police car (siren)","Ambulance (siren)","Fire engine, fire truck (siren)","Car alarm"],
  "rail": ["Rail transport","Train","Train whistle","Train horn","Railroad car, train wagon","Train wheels squealing","Subway, metro, underground"],
  "aircraft": ["Aircraft","Aircraft engine","Jet engine","Propeller, airscrew","Helicopter","Fixed-wing aircraft, airplane"],
  "engine_motion": ["Engine","Light engine (high frequency)","Medium engine (mid frequency)","Heavy engine (low frequency)","Engine knocking","Engine starting","Idling","Accelerating, revving, vroom"],
  "alarms_buzzer": ["Alarm","Alarm clock","Buzzer","Smoke detector, smoke alarm","Fire alarm"],
  "phone_ring": ["Telephone bell ringing","Ringtone","Telephone","Telephone dialing, DTMF","Dial tone","Busy signal"],
  "wind_rain": ["Wind","Rustling leaves","Wind noise (microphone)","Thunderstorm","Thunder","Rain","Raindrop","Rain on surface"],
  "door_knock": ["Door","Doorbell","Ding-dong","Knock","Tap","Slam","Sliding door","Cupboard open or close","Drawer open or close"],
  "glass_break": ["Glass","Shatter","Smash, crash","Breaking","Chink, clink","Crack"],
  "explosion_gunshot": ["Explosion","Gunshot, gunfire","Machine gun","Fusillade","Artillery fire","Cap gun","Fireworks","Firecracker","Burst, pop","Eruption","Boom"],
  "Silence": ["Silence", "Inside, small room", "Inside, large room or hall"],
  "other": "REST"
}

AGGREGATION = "sum"  # or "mean"
EXPORT_DIR = "reduced_yamnet_savedmodel"  # output folder
LABELS_JSON = "reduced_labels.json"       # output labels order

def load_yamnet_names():
    # (1) Preferred: from hub asset
    try:
        m = hub.load("https://tfhub.dev/google/yamnet/1")
        path = m.class_map_path().numpy().decode("utf-8")
        names = []
        with tf.io.gfile.GFile(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["display_name"])
        if len(names) != 521:
            raise ValueError(f"class count {len(names)} != 521")
        return names
    except Exception as e:
        # (2) Local fallback
        local_csv = os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv")
        if tf.io.gfile.exists(local_csv):
            names = []
            with tf.io.gfile.GFile(local_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    names.append(row["display_name"])
            if len(names) != 521:
                raise
            return names
        raise RuntimeError(f"Could not load YAMNet class map: {e}")

def build_pooling_matrix(names, reduced_map, aggregation="sum"):
    K = len(reduced_map)
    tgt_labels = list(reduced_map.keys())
    name2idx = {n: i for i, n in enumerate(names)}
    M = np.zeros((len(names), K), dtype=np.float32)
    assigned = set()
    rest_k = None
    for k, t in enumerate(tgt_labels):
        spec = reduced_map[t]
        if spec == "REST":
            rest_k = k
            continue
        for n in spec:
            if n not in name2idx:
                print(f"[WARN] '{n}' not in YAMNet names; skipping.")
                continue
            j = name2idx[n]
            M[j, k] = 1.0
            assigned.add(j)
    if rest_k is not None:
        for j in range(len(names)):
            if j not in assigned:
                M[j, rest_k] = 1.0
    if aggregation == "mean":
        col = M.sum(axis=0, keepdims=True)
        col[col == 0] = 1.0
        M = M / col
    return M.astype(np.float32), tgt_labels

class ReducedModule(tf.Module):
    def __init__(self, M):
        super().__init__()
        # Use hub.load SavedModel directly (no KerasLayer -> fewer trackables)
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")  # callable: y -> (scores, embeddings, spectrogram)
        self.M = tf.constant(M, tf.float32)  # [521, K]

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32, name="waveform_16k")])
    def __call__(self, waveform_16k):
        # YAMNet expects 1-D float32 at 16 kHz
        scores, embeddings, _ = self.yamnet(waveform_16k)  # [T,521]
        pooled = tf.matmul(scores, self.M)                 # [T,K]
        probs = tf.reduce_mean(pooled, axis=0)             # [K]
        return {"probs": probs}  # named output for SavedModel

def main():
    names = load_yamnet_names()
    M, labels = build_pooling_matrix(names, REDUCED_MAP, AGGREGATION)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    with open(LABELS_JSON, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"[OK] labels -> {LABELS_JSON}: {labels}")

    module = ReducedModule(M)
    # Warmup
    _ = module(tf.zeros([16000], tf.float32))

    # Export SavedModel
    tf.saved_model.save(module, EXPORT_DIR,
                        signatures=module.__call__.get_concrete_function())
    print(f"[OK] SavedModel -> {EXPORT_DIR}")

    # Optional: TFLite (may fail on 3.13; safe to try)
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
        tflite = converter.convert()
        open(os.path.join(EXPORT_DIR, "reduced_yamnet.tflite"), "wb").write(tflite)
        print(f"[OK] TFLite -> {EXPORT_DIR}/reduced_yamnet.tflite")
    except Exception as e:
        print(f"[WARN] TFLite conversion skipped: {e}")

if __name__ == "__main__":
    main()
