Title | Icon | Licence | Badge 1
:---:|:---:|:---:|:---:
MLOps template | 🦄 | MIT | [![Badge example 1](https://github.com/sycod/container_test/actions/workflows/main.yaml/badge.svg)](https://github.com/sycod/container_test/actions/workflows/main.yaml)

# Todo

- [x] xxx
- [ ] dire de faire un fichier ``.env`` et son contenu
- [ ] write api manual & limits
- [ ] ckech TF version
    ⬇️   ⬇️
``` python
python3 -m pip install tensorflow[and-cuda]
# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

TensorBoard usage
terminal in log folder: `tensorboard --logdir logs`