If you don't have a reduced SavedModel, create or download one and place it under:

models/reduced_yamnet_savedmodel/

The pipeline loads the model with `tf.saved_model.load(REDUCED_MODEL_DIR)`.
If you prefer to use full YAMNet from TF-Hub, set `USE_REDUCED = False` in `Pipeline.py`.

You can override the default path with the environment variable `REDUCED_MODEL_DIR`.
