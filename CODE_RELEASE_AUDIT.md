# Code release audit

## Release scope

Minimal reproducible implementation of the formal ungated MGTF-Net main model. Existing private project files and training processes were not modified. No proprietary training checkpoint or scientific benchmark result is part of this release.

## File inventory

- `.gitignore`: excludes local demo checkpoints, caches, logs, results, and environment files
- `README.md`: model, public interface, units, limitations, and quick start
- `assets/mgtf_net_framework.png`: author-provided framework schematic, copied without modification and displayed in the README
- `requirements.txt`: NumPy and PyTorch only
- `config.py`: formal architecture and training configuration
- `model/__init__.py`: main-model exports
- `model/mgtf_net.py`: four-branch model and exact native multi-horizon loss
- `model/graph_attention.py`: scene-isolated, fully connected GATv2 with self edges
- `model/temporal_encoder.py`: formal Transformer initialization
- `model/decoder.py`: continuous autoregressive decoder
- `data/__init__.py`: synthetic-data package
- `data/synthetic_dataset.py`: independent artificial kinematic generator
- `data/example_synthetic.npz`: entirely generated numerical example
- `train_demo.py`: CPU training and validation demonstration
- `inference_demo.py`: CPU prediction and six metric outputs
- `utils/__init__.py`: utility package
- `utils/preprocessing.py`: generic sequence, normalization, indexing, and tensor interfaces
- `utils/metrics.py`: metric definitions in meters
- `utils/seed.py`: deterministic demo initialization
- `CODE_RELEASE_AUDIT.md`: this audit

## Privacy and exclusions

No real-data interface was copied: geographic parsers, airport-specific transforms, source-file readers, data paths, and real identifiers are absent. The release contains no absolute local paths, real airport names, real flight identifiers, proprietary data, data caches, logs, trained checkpoints, experimental results, experimental variants, graph diagnostics, sensitivity studies, plotting scripts, or obsolete model implementations. Documentation explains these exclusions; no excluded implementation is present. Git metadata identifies the hosting account but no account name or private contact details are embedded in the code or data.

The artificial example is produced solely from NumPy random numbers and analytic Cartesian kinematics. Its fields are `observed_features`, `future_positions`, `scene_ids`, `dt`, and `generator_seed`; all are numerical arrays loaded with `allow_pickle=False`. It contains three synthetic scenes, 13 artificial aircraft, 12 observations, 20 future samples, and a 5-second interval. No external file or real-data source is consulted by the generator. The main demo seed is 2022; fixed offsets generate independent validation and example data, not repeated paper experiments.

## Formal implementation parity

The private formal source snapshot and current main-model source had identical SHA-256 digests:

`b36cfdfdeab7b016428c1aed41a1b1966d02e80e2168459bd2533c6f9008afe1`

The release extracts only the formal continuous autoregressive configuration; unused decoder alternatives and compatibility aliases were omitted. Module parameter names, buffer names, shapes, initialization order, normalization convention, forward computation, self edges, fully connected scene graphs, normalized 3D edge features, decoder trend, and loss were retained. The public meter-unit adapter is an interface conversion, not a model change.

No discrepancy was found between the requested architecture dimensions and this formal implementation. `MAX_EPOCHS = 300` is the reference configuration requested for release; a private continuation used a larger epoch budget without changing the model. The demo default is three epochs. No private experiment queue or continuation management is included.

Trainable parameters: **234,691**.

## Verification

Verified CPU runtime: Python with PyTorch **2.11.0+cu128** and NumPy **2.5.1**. CUDA was not used by the demos. Lower dependency bounds are interface requirements, not a claim that every supported version was tested.

The following checks passed:

1. All initialized state-dictionary tensors exactly equal the private formal continuous-AR model when initialized with the same seed.
2. Forward outputs match the formal model bitwise on artificial multi-scene inputs.
3. Native multi-horizon loss and every parameter gradient match bitwise; gradients are finite.
4. Changing another scene leaves the first scene's predictions exactly unchanged.
5. Providing future timesteps to the observation interface raises an error.
6. A known `[3,4,12]` meter error gives 13-meter 3D error, 5-meter horizontal error, and 12-meter vertical error for all corresponding metrics.
7. `python train_demo.py --epochs 2` completes forward, loss, backward, clipping, optimizer updates, validation, and local best-checkpoint saving without errors.
8. `python inference_demo.py` completes with input `[13,12,9]` and predicted positions `[13,20,3]`, and prints six finite meter-unit metrics.

Demo metrics are artificial execution checks only and do not reproduce, replace, or support the paper's reported performance. Checkpoints produced during verification remain ignored local files, not release content.
