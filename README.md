[Uploading README.md…]()
# DCLM-impute

## A Dual Contrastive Learning Framework with Multi-constraint for Imputing Single-cell RNA Sequencing Data

**DCLM-impute** is a dual contrastive learning framework with multi-constraint for imputing single-cell RNA sequencing (scRNA-seq) data. It is designed to address two limitations of existing imputation methods: the difficulty of balancing local and global information, and the tendency of a single condition or constraint to inadequately capture the inherent complexity of biological data.

The framework comprises three key components:

1. **Cell augmentation**
2. **Dual contrastive learning with multi-constraint**
3. **Imputation**

The current repository provides the research implementation and a reproducible example based on the **Zeisel** dataset.

> **Repository status:** research code accompanying the DCLM-impute manuscript. The current entry script is configured for the Zeisel experiment and should be adjusted before application to another dataset.

---

## Overview of DCLM-impute

<p align="center">
  <img src="assets/DCLM-impute_overview.jpg" alt="Overview of DCLM-impute" width="100%">
</p>

**Figure 1. Overview of DCLM-impute.**  
**(A) Cell augmentation.** The scRNA-seq count matrix \(R\) is converted into the raw expression matrix \(X \in \mathbb{R}^{n \times m}\) after gene filtering, where \(n\) and \(m\) denote the numbers of cells and genes, respectively. K-means clustering is applied to \(X\) to generate **pre-labels**. Random masking is then used to construct three cell augmentation views \(x_1\), \(x_2\), and \(x_3\).  
**(B) Dual contrastive learning with multi-constraint.** The framework contains two complementary branches. In **contrastive learning with multi-constraint (CC)**, \(x_1\) and \(x_2\) are encoded by multi-head self-attention to obtain \(y_1\) and \(y_2\). The CC branch is jointly constrained by the contrastive loss, ZINB autoencoder loss, bottleneck classifier loss, and reconstruction loss to capture global dependencies and biologically meaningful representations. In **contrastive learning with multi-scale self-attention mechanism (CS)**, \(x_3\) is processed by a multi-scale convolutional layer and self-attention to obtain \(y_3\). A negative-free self-distillation strategy with online and target networks is used to capture local expression patterns. The representations \(y_1\), \(y_2\), and \(y_3\) are integrated through **features fusion** to obtain the unified latent representation \(C\).  
**(C) Imputation.** A K-nearest neighbor (KNN) graph is constructed in the latent space defined by \(C\). Least-squares regression is used to estimate the association weights between each target cell and its neighboring cells. Nonzero values are preserved, whereas predicted values are used to recover dropout zeros, yielding the final imputed matrix \(X'\).

---

## Method

### A. Cell augmentation

The cell augmentation stage performs three operations:

1. **Gene filtering:** low-quality genes are removed from the scRNA-seq count matrix \(R\), producing the raw expression matrix \(X\).
2. **Pre-label generation:** K-means clustering is applied to \(X\), and the resulting cluster assignments are used as pre-labels.
3. **Random masking:** independently sampled random masking functions are applied to nonzero values to generate three augmented views \(x_1\), \(x_2\), and \(x_3\).

### B. Dual contrastive learning with multi-constraint

#### B1. Contrastive learning with multi-constraint (CC)

The CC branch receives \(x_1\) and \(x_2\). A multi-head self-attention encoder learns global intercellular dependencies and produces the representations \(y_1\) and \(y_2\).

The total optimization objective contains four constraints:

- **contrastive loss**;
- **ZINB autoencoder loss**;
- **bottleneck classifier loss**;
- **reconstruction loss**.

The default loss weights implemented in the current code are:

```text
0.4 × contrastive loss
0.3 × bottleneck classifier loss
0.2 × ZINB autoencoder loss
0.1 × reconstruction loss
```

#### B2. Contrastive learning with multi-scale self-attention mechanism (CS)

The CS branch receives \(x_3\). It uses:

- a multi-scale one-dimensional convolutional layer to extract local expression patterns;
- a self-attention layer to model dependencies among the extracted features;
- an online network and a target network for negative-free self-distillation;
- an exponential moving average to update the target network.

The CS branch produces the representation \(y_3\).

#### B3. Features fusion

The representations learned by CC and CS are fused into a unified latent representation \(C\). The current implementation uses a fusion weight of `0.25` for the CS representation.

### C. Imputation

For each target cell:

1. KNN is used to identify neighboring cells in the latent space of \(C\);
2. least-squares regression estimates the association weights between the target cell and its neighbors;
3. neighboring expression profiles are linearly aggregated to predict missing expression values;
4. nonzero values in \(X\) are preserved, while zero values are replaced by the corresponding predictions.

The output is the imputed expression matrix \(X'\).

---

## Paper terminology and code mapping

The following table keeps the repository terminology strictly aligned with the manuscript.

| Manuscript term | Symbol | Current code |
|---|---|---|
| scRNA-seq count matrix | \(R\) | input CSV matrix |
| raw expression matrix | \(X\) | `groundTruth_data` / processed input |
| pre-labels | — | `pre_label`, `pre_label_tensor` |
| cell augmentation views | \(x_1,x_2,x_3\) | augmented tensors produced by `data_augmentations()` |
| contrastive learning with multi-constraint | CC | `training_simclr()` and `SelfAttention` |
| multi-head self-attention representations | \(y_1,y_2\) | hidden representations from the CC branch |
| contrastive loss | \(\mathcal{L}_{contrastive}\) | `ConstrastiveLoss` |
| ZINB autoencoder loss | \(\mathcal{L}_{ZINB}\) | `ZINBLoss` |
| bottleneck classifier loss | \(\mathcal{L}_{classifier}\) | classification loss in `training_simclr()` |
| reconstruction loss | \(\mathcal{L}_{reconstruction}\) | reconstruction MSE in `training_simclr()` |
| contrastive learning with multi-scale self-attention mechanism | CS | `train_byol()`, `AttentionEncoder`, and `BYOLModel` |
| CS representation | \(y_3\) | hidden representation from the online encoder |
| features fusion | \(C\) | `hidden_states_new` |
| K-nearest neighbor graph | KNN | `select_neighbours()` |
| least-squares imputation | — | `LS_imputation()` |
| imputed expression matrix | \(X'\) | `imputed_data` |

> The class and function names are retained for compatibility with the current source code. In the manuscript and documentation, the two branches should be referred to as **CC** and **CS**, rather than “SimCLR branch” and “BYOL branch”.

---

## Repository structure

```text
DCLM-impute/
├── main.py                  # Complete DCLM-impute execution pipeline
├── models.py                # CC and CS encoders, online network, and target network
├── training.py              # Training procedures for CC and CS
├── losses.py                # Contrastive, NB, ZINB, and cosine losses
├── augmentations.py         # Random masking for cell augmentation
├── preprocessing.py         # Data loading, filtering, normalization, and pre-label generation
├── imputation.py            # KNN construction and least-squares imputation
├── assets/
│   └── DCLM-impute_overview.jpg
├── docs/
│   ├── USAGE_GUIDE_CN.md
│   └── RELEASE_CHECKLIST_CN.md
├── Zeisel_top2000.csv       # Example expression matrix
├── Zeisel_cell_label.csv    # Zeisel cell annotations
└── requirements.txt
```

## Requirements

A suggested environment is:

- Python 3.9–3.11
- PyTorch 2.0 or later
- NumPy
- pandas
- scikit-learn

Install the dependencies with:

```bash
pip install -r requirements.txt
```

For a CUDA-enabled installation of PyTorch, use the installation command corresponding to the local CUDA version.

## Installation

```bash
git clone https://github.com/iwasto/DCLM-impute.git
cd DCLM-impute

python -m venv .venv
```

Activate the environment.

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Input data format

The input expression matrix must be a CSV file with:

- genes in rows;
- cells in columns;
- the first column containing gene identifiers;
- nonnegative expression or count values in the remaining entries.

Example:

```text
gene,cell_1,cell_2,cell_3
GeneA,0,3,1
GeneB,5,0,2
GeneC,0,1,0
```

`preprocessing.load_data()` reads the file and internally converts the matrix to the cell-by-gene orientation used by the model.

## Quick start

### 1. Correct the example data path

The current `main.py` contains:

```python
groundTruth_data, cells, genes = load_data("Zeisel")
```

Because `load_data()` expects a CSV path, change it to:

```python
groundTruth_data, cells, genes = load_data(
    "Zeisel_top2000.csv"
)
```

A recommended configuration block is:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset_name = "Zeisel"
data_path = "Zeisel_top2000.csv"
drop_rate = 0.1
seed = 42

groundTruth_data, cells, genes = load_data(data_path)
drop_data = impute_dropout(
    groundTruth_data,
    seed=seed,
    drop_rate=drop_rate,
)
```

### 2. Run the example

```bash
python main.py
```

The script performs the following steps:

1. load the scRNA-seq expression matrix;
2. generate an artificial dropout matrix for evaluation;
3. perform normalization and generate pre-labels;
4. generate the cell augmentation views \(x_1\), \(x_2\), and \(x_3\);
5. train the CC branch to obtain \(y_1\) and \(y_2\);
6. train the CS branch to obtain \(y_3\);
7. perform features fusion to obtain \(C\);
8. construct the KNN graph in the latent space;
9. perform least-squares imputation and save \(X'\).

## Main parameters

| Parameter | Current value | Manuscript meaning |
|---|---:|---|
| `drop_rate` | `0.1` | Artificial dropout rate used for recovery evaluation |
| `n_cluster` | `7` | Number \(K\) of clusters used to generate pre-labels |
| `hidden_size` | `256` | Dimension of the learned representations |
| `epochs` | `100` | Number of training epochs for CC and CS |
| `aug_rate` | `0.4` | Random masking rate used to construct cell augmentation views |
| `k` | `20` | Neighbor size used to construct the KNN graph |
| CS fusion weight | `0.25` | Weight of the CS representation in features fusion |
| `filter_noise` | `2` | Threshold used to remove low predicted expression values |

## Applying DCLM-impute to another dataset

To apply the method to another dataset:

1. prepare a gene-by-cell CSV matrix;
2. set `data_path` to the new file;
3. set `n_cluster` to the expected number of cell groups;
4. update the bottleneck classifier output dimension in `models.py`:

```python
nn.Linear(hidden_size, number_of_clusters)
```

5. adjust the random masking rate, neighbor size, fusion weight, hidden size, and training epochs when necessary;
6. reduce the number of cells or adopt memory-efficient attention for very large datasets.

The current multi-head self-attention implementation constructs a cell-by-cell attention matrix; therefore, its memory use increases approximately quadratically with the number of cells.

## Reproducibility

For deterministic or approximately reproducible experiments, add the following before data processing and training:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The artificial dropout function accepts a seed:

```python
drop_data = impute_dropout(
    groundTruth_data,
    seed=seed,
    drop_rate=drop_rate,
)
```

## Output orientation

The model internally uses a cell-by-gene matrix. The saved matrices are transposed back to gene-by-cell orientation:

```python
pd.DataFrame(
    imputed_data.T,
    index=genes,
    columns=cells,
)
```

Recommended output names:

```python
dropout_path = (
    f"{dataset_name}_dropout_{drop_rate:.1f}.csv"
)
imputed_path = (
    f"{dataset_name}_DCLM_imputed_{drop_rate:.1f}.csv"
)
```

## Important implementation notes

- The current example is configured for seven pre-label clusters.
- The bottleneck classifier in `models.py` is fixed to seven output classes.
- The current `main.py` uses the CPU by default.
- The example output filename `impute_0.4.csv` is inconsistent with the default `drop_rate=0.1`.
- If the least-squares system is singular, the current implementation switches to average-based imputation.
- The repository currently does not provide a command-line interface or automatic dataset configuration.
- The current `__init__.py` contains notebook metadata rather than Python package initialization code.

Detailed Chinese instructions are provided in [`docs/USAGE_GUIDE_CN.md`](docs/USAGE_GUIDE_CN.md).

## Citation

Please cite the associated manuscript when using this code:

```text
Fang H, Li G, Huang S, et al.
DCLM-impute: A Dual Contrastive Learning Framework with Multi-constraint
for Imputing Single-cell RNA Sequencing Data.
```

A BibTeX entry can be added after the final publication information is available.

## Contact

For questions concerning the implementation, experiments, or manuscript, please open a GitHub issue or contact the corresponding authors.

## License

A software license has not yet been specified in the current repository. Add an explicit license before formal public distribution or reuse.
