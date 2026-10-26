# TrafficModelMar

Traffic forecasting experiments built on
[BasicTS](https://github.com/GestaltCogTeam/BasicTS) [1].

## Layout

```
baseline/
├── PEMS03.py  PEMS04.py  PEMS07.py  PEMS08.py    # one config per dataset
├── CA.py  GBA.py  GLA.py  SD.py
└── arch/
    ├── margst.py    # MargST: the model
    └── losses.py    # masked metrics + trend loss
```

## Model

MargST predicts on the **margins** of the spatio-temporal tensor instead of on the
full grid. The node-mean and time-mean marginals are encoded separately by two
light MLP stacks — ViT-style blocks from `timm` [2, 3] with pre-LN residuals [4] —
fused multiplicatively into a `(B, L, N, D)` context, blended with the embedded
input through two learnable scalars, and mapped to the horizon by a residual MLP
head.

Inputs carry time-of-day, day-of-week and a learnable spatio-temporal adaptive
embedding, following STAEformer [5]. Metrics use the masked convention of
DCRNN [6]; the loss adds a first-order difference term to masked MAE.

## Datasets

All data is distributed by BasicTS [1].

| Config | Nodes | Resolution | Steps | Source |
| --- | ---: | ---: | ---: | --- |
| PEMS03 / 04 / 07 / 08 | 358 / 307 / 883 / 170 | 5 min | 16,992–28,224 | PEMS0X [7] |
| CA / GBA / GLA / SD | 8,600 / 2,352 / 3,834 / 716 | 15 min | 35,040 | LargeST [8] |

Input/output length and split ratios are not hard-coded: they are read from
BasicTS through `get_regular_settings()` [1].

## Training

12 → 12 steps · Adam, lr 5e-4 · MultiStepLR, milestones [30, 50, 70], γ=0.1 ·
batch 32 · 100 epochs · early stopping, patience 20 · masked MAE / MAPE / RMSE,
MAE as the selection target.

## Usage

Requires a **pre-1.0** BasicTS. The configs import
`TimeSeriesForecastingDataset`, `SimpleTimeSeriesForecastingRunner` and
`basicts.metrics.masked_*`, all renamed or hidden in 1.0 (Oct 2025);
commit `4c3e2d3` (2025-05-26) is verified to expose every symbol used here.

```bash
git clone https://github.com/GestaltCogTeam/BasicTS.git && cd BasicTS
git checkout 4c3e2d3
pip install -r requirements.txt && pip install timm

cp -r /path/to/baseline baselines/MargST
python experiments/train.py    -c   baselines/MargST/PEMS04.py -g 0
python experiments/evaluate.py -cfg  baselines/MargST/PEMS04.py \
    -ckpt checkpoints/MargST/PEMS04_100_12_12/<RUN_ID>/MargST_best_val_MAE.pt -g 0
```

The directory must sit two levels below the BasicTS root, because the configs do
`sys.path.append(os.path.abspath(__file__ + '/../../..'))`.

## References

**[1]** Z. Shao et al. *Exploring Progress in Multivariate Time Series Forecasting:
Comprehensive Benchmarking and Heterogeneity Analysis.* IEEE TKDE 37(1):291–305,
2024. <https://github.com/GestaltCogTeam/BasicTS>

**[2]** A. Dosovitskiy et al. *An Image Is Worth 16×16 Words: Transformers for
Image Recognition at Scale.* ICLR 2021.

**[3]** R. Wightman. *PyTorch Image Models* (`timm`). 2019.
doi:10.5281/zenodo.4414861

**[4]** R. Xiong et al. *On Layer Normalization in the Transformer Architecture.*
ICML 2020.

**[5]** H. Liu et al. *Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer
SOTA for Traffic Forecasting.* CIKM 2023. arXiv:2308.10425

**[6]** Y. Li et al. *Diffusion Convolutional Recurrent Neural Network: Data-Driven
Traffic Forecasting.* ICLR 2018. arXiv:1707.01926

**[7]** S. Guo et al. *Learning Dynamics and Heterogeneity of Spatial-Temporal Graph
Data for Traffic Forecasting.* IEEE TKDE 34(11):5415–5428, 2022.

**[8]** X. Liu et al. *LargeST: A Benchmark Dataset for Large-Scale Traffic
Forecasting.* NeurIPS 2023, Datasets and Benchmarks Track. arXiv:2306.08259

## License

No open-source license is attached; all rights reserved. `arch/losses.py` is
adapted from the author's own `Traffic_Model` code. The papers cited above remain
under their own licenses.
