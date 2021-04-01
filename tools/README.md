# Tools description

### test.py

```bash
python tools/train.py
```

Option | Description
--- | ---
`--config [config path]` | Optional config file path. The `configs/default.yaml` is loaded by default. The specified file overwrites the default configs.
`--out-dir [outdir path]`  | Output directory path. (default: `results`)
`--resume [ckpt path]` | Resuming checkpoint file path. 

### train.py

```bash
python tools/test.py [ckpt path]
```

Option | Description
--- | ---
`--config [config path]` | The optional config file path.used when training.
`--show-dir [outdir path]`  | Path to save predict visualization. Please specify if you want to save.


If you want to override the config with command line args, put them at the end in the form of dotlist.

```bash
python tools/train.py --config [config path] SOLVER.NUM_WORKERS=8 SOLVER.EPOCH=5
```

### visualize_nyu2_test_gt.py

Visualize the dataset gt as well as visualizations of prediction results for comparison.

```bash
python tools/visualize_nyu2_test_gt.py
```

Option | Description
--- | ---
`--outdir [outdir path]`  | Output directory path. (default: `vis_gt`)
`--config [config path]` | Optional config file path.
`--debug [ckpt path]` | Use Debug mode. (visualize just one data)
