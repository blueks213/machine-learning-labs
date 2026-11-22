# 机器学习实验

本科《机器学习》课程的三次实验代码。数据文件较大时未纳入仓库，按下方说明下载后放到仓库根目录即可运行。

## 实验列表

| 文件 | 内容 | 数据 |
| --- | --- | --- |
| `1.py` | 红酒质量分类：逻辑回归 / 随机森林、网格搜索、混淆矩阵 | `winequality-red.csv`（已包含） |
| `2.py` | 洛杉矶犯罪数据回归：线性回归 / 随机森林 / SVR，时间特征工程 | `Crime_Data_from_2020_to_Present.csv`（需下载） |
| `3.py` | 在线零售客户价值分析：RFM、PCA、KMeans / DBSCAN / 层次聚类 | `Online Retail.xlsx`（需下载） |

## 数据下载

- 犯罪数据（Los Angeles Open Data，约 195 MB）：  
  <https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8>
- 在线零售数据（UCI Machine Learning Repository，约 23 MB）：  
  <https://archive.ics.uci.edu/dataset/352/online+retail>
- 红酒质量数据（UCI，已包含）：  
  <https://archive.ics.uci.edu/dataset/186/wine+quality>

下载后请保持文件名与代码中的文件名一致，并放在仓库根目录。

## 运行

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python 1.py
```

`1.py`、`3.py` 会弹窗显示图表；无图形界面环境可把 `plt.show()` 改为 `plt.savefig(...)`。

## 说明

- 原始实验报告、指导书和打包压缩包未纳入仓库。
- 代码保留课程实验原貌，主要价值在于特征工程与聚类/分类流程对比。
