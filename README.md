# TrafficModelMar

Research repository for traffic forecasting experiments. Every model in
this repository is packaged as a drop-in module for
[BasicTS](https://github.com/GestaltCogTeam/BasicTS) [1], which supplies the data
pipeline, the runner, the scaler and the evaluation metrics.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Model: MargST](#model-margst)
  - [Pipeline](#pipeline)
  - [Component provenance](#component-provenance)
  - [Input encoding](#input-encoding)
  - [Marginal fusion](#marginal-fusion)
- [Datasets](#datasets)
- [Training setup](#training-setup)
- [Requirements and usage](#requirements-and-usage)
- [References](#references)
- [License and attribution](#license-and-attribution)

---

## Repository layout

```
baseline/
├── __init__.py            # empty; makes the directory a package
├── CA.py                  # one complete BasicTS config per dataset
├── GBA.py
├── GLA.py
├── SD.py
├── PEMS03.py
├── PEMS04.py
├── PEMS07.py
├── PEMS08.py
└── arch/
    ├── __init__.py        # re-exports MargST
    ├── margst.py          # the model
    └── losses.py          # masked MAE/MAPE/RMSE + masked_trend_loss
```

Each `*.py` at the top level is a *self-contained training configuration*: it
imports the model, sets `MODEL_PARAM`, and fills in `CFG.DATASET`,
`CFG.SCALER`, `CFG.METRICS`, `CFG.TRAIN`, `CFG.VAL` and `CFG.TEST` following the
BasicTS config protocol [1].

---

## Model: MargST

MargST is a *marginal* spatio-temporal forecaster. Instead of encoding the full
`(time × node)` grid, it summarises the input along each axis separately,
encodes the two marginals with cheap per-token MLPs, fuses them back into the
grid, and predicts with a residual MLP stack. The name refers to working on the
margins of the spatio-temporal tensor rather than on the joint tensor.

The implementation in `arch/margst.py` is a clean reference version: it keeps
only the final architecture and drops the ablation switches (encoder type,
aggregation mode, fusion variants, STAR core) that were used during
development.

### Pipeline

```
x  (B, L, N, C)
  -> input encoding + TOD/DOW/adaptive embeddings       (B, L, N, D)
  -> U_T = TemporalEncoder(mean_n x)                    (B, L, 1, D)
     U_S = SpatialEncoder(mean_t x)                     (B, 1, N, D)
  -> Z = U_T * U_S                                      (B, L, N, D)
  -> H = softplus(alpha) * Z + softplus(beta) * x       (B, L, N, D)
  -> merge time into channels, residual MLP stack
  -> output projection                                  (B, L_out, N, C_out)
```

`alpha` and `beta` are scalar learnables initialised so that
`softplus(alpha) = 0.693` and `softplus(beta) = 1.313`, i.e. the fused
representation starts closer to the encoded input than to the marginal context.

### Component provenance

Every design choice below is borrowed or re-derived; the right column points to
the paper or implementation it comes from.

| Component | Where it lives | Origin |
| --- | --- | --- |
| Time-of-day / day-of-week embeddings | `MargST._create_embeddings` | Periodicity embedding `E_p` of STAEformer [2] |
| Learnable spatio-temporal adaptive embedding, shared across the batch | `MargST.adaptive_embedding` | Adaptive embedding `E_a` of STAEformer [2]; node-adaptive parameters of AGCRN [3] |
| MLP block (Linear → GELU → Dropout → Linear) | `MarginalEncoder` via `timm` | The MLP block of ViT [4], as implemented in `timm` [5] |
| Pre-LN residual block `x = x + f(LayerNorm(x))` | `MarginalEncoder.forward`, `MargST.forward` | Pre-LN Transformer [6]; pre-activation ResNet residual form |
| Masked MAE / MAPE / RMSE with mask renormalisation | `basicts.metrics` [1], mirrored in `arch/losses.py` | DCRNN [7] |
| First-order difference (trend) penalty | `masked_trend_loss` | Author's own `Traffic_Model` code; conceptually related to trend/seasonality decomposition [8] |
| Data pipeline, scaling, runner, config protocol | all `*.py` configs | BasicTS [1] |

The marginal-fusion formulation itself (mean over nodes → temporal context, mean
over time → spatial context, multiplicative fusion, scalar gating) is specific
to this repository; no prior work is claimed for it.

### Input encoding

`DataEncoding` concatenates several views of the same input window and projects
them back to `input_embedding_dim`:

```
raw value │ recent means over windows (3, 6, 12) │ latest value │ first-order difference
```

The multi-scale recent means and the difference branch are cheap hand-crafted
features that give the following MLP stack an explicit view of local level and
local slope, which is useful because no attention layer is used anywhere in this
model.

### Marginal fusion

```python
u_t = temporal_encoder(embedded.mean(dim=2))   # average over nodes   -> (B, L, 1, D)
u_s = spatial_encoder (embedded.mean(dim=1))   # average over time    -> (B, 1, N, D)
context = u_t * u_s                            # broadcast to        -> (B, L, N, D)
h = softplus(alpha) * context + softplus(beta) * embedded
```

Both encoders are `MarginalEncoder` instances — a stack of pre-LN residual MLP
blocks [4, 6]. Because each marginal carries a single sequence per batch element,
the encoders perform channel mixing only: there is no cross-token interaction, and
the cost is `O(L·D² + N·D²)` instead of the `O(L·N·D²)` of encoding the full grid.
The multiplicative fusion lets a node-timeslot pair be represented by the product
of "what usually happens at this time" and "what usually happens at this node",
which is the same factorisation idea behind low-rank spatio-temporal models.

---

## Datasets

Eight datasets are configured. All of them are distributed by BasicTS [1].

| Config | Nodes | Time steps/day | Span | Source |
| --- | ---: | ---: | ---: | --- |
| `PEMS03.py` | 358 | 288 | 26,208 | PEMS0X, via ASTGNN [9] |
| `PEMS04.py` | 307 | 288 | 16,992 | PEMS0X, via ASTGNN [9] |
| `PEMS07.py` | 883 | 288 | 28,224 | PEMS0X, via ASTGNN [9] |
| `PEMS08.py` | 170 | 288 | 17,856 | PEMS0X, via ASTGNN [9] |
| `CA.py` | 8,600 | 96 | 35,040 | LargeST, California [10] |
| `GBA.py` | 2,352 | 96 | 35,040 | LargeST, Greater Bay Area [10] |
| `GLA.py` | 3,834 | 96 | 35,040 | LargeST, Greater London Area [10] |
| `SD.py` | 716 | 96 | 35,040 | LargeST, San Diego [10] |

`PEMS0X` runs at a 5-minute resolution; the four LargeST subsets run at 15
minutes. Input/output lengths and train/val/test ratios are not hard-coded:
they are read from BasicTS through `get_regular_settings(DATA_NAME)` [1], so the
configs stay in sync with the benchmark defaults.

---

## Training setup

Identical across all eight configs.

| Setting | Value |
| --- | --- |
| Input / output length | 12 / 12 |
| Model width `D` | 96 (`input`/`tod`/`dow`/`adaptive` = 24 each) or 128 (32 each) |
| Marginal encoder depth | 2 pre-LN MLP blocks, ratio 2.0, dropout 0.1 |
| Final MLP stack depth | 4 (PEMS03/04/07/08, SD) or 6 (CA, GBA, GLA) |
| Adaptive-embedding dropout | 0.5 |
| Loss | `masked_trend_loss(lambda_trend=1.0)` = masked MAE + first-order-difference MAE |
| Optimiser | Adam, lr 5e-4, weight decay 0 |
| LR schedule | MultiStepLR, milestones `[30, 50, 70]`, gamma 0.1 |
| Early stopping | patience 20 |
| Batch size | 32 for train / val / test |
| Epochs | 100 |
| Metrics | masked MAE, MAPE, RMSE; MAE is the selection target |
| Evaluation horizons | 1–12 |

---

## Requirements and usage

Beyond BasicTS itself the code only needs `torch`, `numpy`, `easydict` and
`timm` (for the MLP block [5]).

> **Version note.** These configs are written against the **pre-1.0** BasicTS
> API. They import `TimeSeriesForecastingDataset`, `SimpleTimeSeriesForecastingRunner`
> and `basicts.metrics.masked_*`, all of which were renamed or hidden in BasicTS
> 1.0 (Oct 2025): the equivalents are `BasicTSForecastingDataset`,
> `BasicTSRunner`, and submodule-level metric imports (`basicts.metrics` no longer
> re-exports anything). Pin a pre-1.0 commit — `4c3e2d3` (2025-05-26) is verified
> to expose every symbol used here — or port the import lines.

The package must sit exactly **two levels below the BasicTS repository root**,
because of `sys.path.append(os.path.abspath(__file__ + '/../../..'))`. The
directory name is free — `baselines/MargST/` matches the upstream convention of
`baselines/<Model>/<Dataset>.py`:

```bash
git clone https://github.com/GestaltCogTeam/BasicTS.git
cd BasicTS
git checkout 4c3e2d3
pip install -r requirements.txt && pip install timm

cp -r /path/to/baseline baselines/MargST
python experiments/train.py -c baselines/MargST/PEMS04.py -g 0
```

`-c/--cfg` takes any of the eight configs; `-g/--gpus` selects visible GPUs.
Checkpoints are written to
`checkpoints/MargST/<DATASET>_<epochs>_<L>_<H>/`.

Evaluation on the test split expects both the config and a checkpoint path
(note that `evaluate.py` spells the flag `-cfg`, not `-c`):

```bash
python experiments/evaluate.py \
    -cfg  baselines/MargST/PEMS04.py \
    -ckpt checkpoints/MargST/PEMS04_100_12_12/<RUN_ID>/MargST_best_val_MAE.pt \
    -g 0
```

---

## References

**[1]** BasicTS — *Exploring Progress in Multivariate Time Series Forecasting:
Comprehensive Benchmarking and Heterogeneity Analysis.* Zezhi Shao, Fei Wang,
Yongjun Xu, Wei Wei, Chengqing Yu, Zhao Zhang, Di Yao, Tao Sun, Guangyin Jin,
Xin Cao, et al. IEEE TKDE, 37(1):291–305, 2024.
Code: <https://github.com/GestaltCogTeam/BasicTS>

**[2]** *Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for
Traffic Forecasting.* Hangchen Liu, Zheng Dong, Renhe Jiang, Jiewen Deng,
Jinliang Deng, Quanjun Chen, Xuan Song. CIKM 2023, pp. 4125–4129.
arXiv:2308.10425 · Code: <https://github.com/XDZhelheim/STAEformer>

**[3]** *Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting.*
Lei Bai, Lina Yao, Can Li, Xianzhi Wang, Can Wang. NeurIPS 2020, pp. 17804–17815.

**[4]** *An Image Is Worth 16×16 Words: Transformers for Image Recognition at
Scale.* Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. ICLR 2021.

**[5]** *PyTorch Image Models* (`timm`). Ross Wightman. 2019.
<https://github.com/huggingface/pytorch-image-models> ·
doi:10.5281/zenodo.4414861

**[6]** *On Layer Normalization in the Transformer Architecture.* Ruibin Xiong,
Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan
Lan, Liwei Wang, Tie-Yan Liu. ICML 2020, pp. 7532–7541.

**[7]** *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
Forecasting.* Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu. ICLR 2018.
arXiv:1707.01926

**[8]** *Autoformer: Decomposition Transformers with Auto-Correlation for
Long-Term Series Forecasting.* Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng
Long. NeurIPS 2021, pp. 22419–22430.

**[9]** *Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for
Traffic Forecasting.* Shengnan Guo, Youfang Lin, Huaiyu Wan, Xiucheng Li, Gao
Cong. IEEE TKDE, 34(11):5415–5428, 2022 (early access 2021).
doi:10.1109/TKDE.2021.3056502 ·
Data: <https://github.com/guoshnBJTU/ASTGNN/tree/main/data>

**[10]** *LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting.* Xu
Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang,
Zhenguang Liu, Bryan Hooi, Roger Zimmermann. NeurIPS 2023, Datasets and
Benchmarks Track. arXiv:2306.08259 · Code: <https://github.com/liuxu77/LargeST>

```bibtex
@article{shao2024exploring,
  title   = {Exploring Progress in Multivariate Time Series Forecasting:
             Comprehensive Benchmarking and Heterogeneity Analysis},
  author  = {Shao, Zezhi and Wang, Fei and Xu, Yongjun and Wei, Wei and Yu, Chengqing
             and Zhang, Zhao and Yao, Di and Sun, Tao and Jin, Guangyin and Cao, Xin
             and others},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {37}, number = {1}, pages = {291--305}, year = {2024}
}

@inproceedings{liu2023spatio,
  title     = {Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA
               for Traffic Forecasting},
  author    = {Liu, Hangchen and Dong, Zheng and Jiang, Renhe and Deng, Jiewen
               and Deng, Jinliang and Chen, Quanjun and Song, Xuan},
  booktitle = {Proceedings of the 32nd ACM International Conference on
               Information and Knowledge Management},
  pages     = {4125--4129}, year = {2023}
}

@inproceedings{liu2023largest,
  title     = {LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting},
  author    = {Liu, Xu and Xia, Yutong and Liang, Yuxuan and Hu, Junfeng and
               Wang, Yiwei and Bai, Lei and Huang, Chao and Liu, Zhenguang and
               Hooi, Bryan and Zimmermann, Roger},
  booktitle = {Advances in Neural Information Processing Systems 36
               (NeurIPS 2023), Datasets and Benchmarks Track},
  year      = {2023}
}
```

---

## License and attribution

No open-source license is attached. The code is publicly readable for reference
and review, but all rights are reserved by the author — please open an issue
before reusing any part of it.

`arch/losses.py` is adapted from the author's `Traffic_Model` code and keeps the
original loss logic, rewritten to the BasicTS keyword signature
(`prediction` / `target` / `null_val`) [1]. Everything else in `arch/` is original
work that builds on the components cited above; the corresponding papers remain
under their own licenses.
