# TrafficModelMar

交通流量预测模型实验（私有仓库）。

## 目录结构

```
baseline/
├── __init__.py
├── CA.py            # CA 数据集配置（California）
├── GBA.py           # GBA 数据集配置（Greater Bay Area）
├── GLA.py           # GLA 数据集配置（Greater London Area）
├── SD.py            # SD 数据集配置（San Diego）
├── PEMS03.py        # PEMS03 配置（358 节点）
├── PEMS04.py        # PEMS04 配置（307 节点）
├── PEMS07.py        # PEMS07 配置（883 节点）
├── PEMS08.py        # PEMS08 配置（170 节点）
└── arch/
    ├── __init__.py
    ├── margst.py    # MargST 模型实现
    └── losses.py    # masked MAE / MAPE / RMSE + masked_trend_loss
```

## MargST 模型

单尺度边缘时空建模（marginal spatio-temporal），不含消融开关的精简参考实现：

```
x (B, L, N, C)
  -> 输入编码 + TOD/DOW/自适应嵌入               (B, L, N, D)
  -> U_T = TemporalEncoder(mean_n x)             (B, L, 1, D)
     U_S = SpatialEncoder(mean_t x)              (B, 1, N, D)
  -> C = U_T * U_S                               (B, L, N, D)
  -> H = softplus(alpha)*C + softplus(beta)*x    (B, L, N, D)
  -> 时间维并入通道 + 残差 MLP 堆叠
  -> 输出投影                                    (B, L_out, N, C_out)
```

依赖 `timm.models.vision_transformer.Mlp`。

## 使用方式

每个 `*.py` 是一份完整的 BasicTS 训练配置（`CFG` + `MODEL_ARCH` + `MODEL_PARAM`），
供 [BasicTS](https://github.com/GestaltCogTeam/BasicTS) 加载。

代码中的 `sys.path.append(os.path.abspath(__file__ + '/../../..'))` 与相对导入
`from .arch import MargST` 要求本目录作为包放在 BasicTS 的模型目录下，例如：

```bash
BasicTS/basicts/model/baseline/
```

随后用 BasicTS 的 runner 指定配置运行即可。

## 训练设置（各数据集一致）

| 项目 | 值 |
|---|---|
| 输入/输出长度 | 12 / 12 |
| 损失函数 | `masked_trend_loss(lambda_trend=1.0)` |
| 优化器 | Adam，lr=5e-4，weight_decay=0 |
| 学习率调度 | MultiStepLR，milestones=[30,50,70]，gamma=0.1 |
| 早停 | patience=20 |
| Batch size | 32（train/val/test 同） |
| 评估 | horizons 1–12 |

## License

私有项目，未授权请勿外传。
