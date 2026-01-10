Place your reduced YAMNet SavedModel and labels here.

Expected structure (DEFAULT):

models/
├─ reduced_yamnet_savedmodel/   <-- SavedModel directory (tf.saved_model)
└─ reduced_labels.json          <-- JSON list of label strings

You can override with environment variables `REDUCED_MODEL_DIR` and `REDUCED_LABELS_JSON`.
