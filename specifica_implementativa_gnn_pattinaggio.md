# Implementation Specification

## Siamese GATv2 for Element Quality Assessment in Figure Skating

**Version:** 1.0\
**Date:** November 11, 2026\
**Project Type:** Academic MVP --- Big Data and Deep Learning

------------------------------------------------------------------------

## 1. Purpose and Objectives

The project aims to build an *Action Quality Assessment* (AQA) system
for figure skating. The system receives the three-dimensional skeleton
of an athlete during a technical element and estimates how similar the
observed pose is to high-quality reference executions.

The chosen approach is a Siamese graph neural network based on
**GATv2**, trained with **Triplet Loss**. The network learns a latent
representation of the skeleton; the distance between the embeddings of
the observed execution and a reference serves as the primary measure of
dissimilarity.

The project is an MVP: it must demonstrate the feasibility of the
end-to-end pipeline on a restricted set of skating elements, not build
an official judging system usable in competition.

### 1.1 Specific Objectives

1.  Retrieve and analyze skating datasets with pre-extracted skeletons
    and quality annotations.
2.  Define a common format for skeletons and their annotations.
3.  Select a small set of sufficiently represented technical elements.
4.  Canonicalize 3D coordinates to reduce differences due to
    translation, scale, and orientation.
5.  Convert each pose into a graph with consistent node topology and
    semantics.
6.  Train a Siamese GATv2 with graph triplets.
7.  Evaluate whether the learned distance is consistent with the
    annotated quality.
8.  Visualize the results and, where possible, the network's attention
    weights.

### 1.2 Out of Scope

The following are outside the scope of the MVP:

-   complete assessment of a full skating program;
-   recognition of all technical elements;
-   3D reconstruction from multi-camera video;
-   automatic generation of an official ISU score;
-   real-time inference;
-   training a new pose estimator;
-   demonstrated generalization across all cameras, filming conditions,
    and athlete categories.

------------------------------------------------------------------------

## 2. Consolidated Design Decisions

  -----------------------------------------------------------------------
  Aspect                              Decision
  ----------------------------------- -----------------------------------
  Sport                               Figure skating

  Primary dataset                     FineFS

  Supplementary dataset               MMFS, pending compatibility
                                      verification

  Representation                      17-joint 3D skeleton

  Academic data pose estimator        HRNet, according to dataset
                                      descriptions

  Topology                            Canonical graph consistent with the
                                      17 joints

  Model                               Siamese GATv2

  Edge features                       Yes, via `edge_attr`

  Training objective                  Metric learning

  Loss                                Triplet Margin Loss with adaptive
                                      margin

  Quality signal                      GOE, when available and
                                      associable with the individual
                                      element

  Primary output                      Distance between embeddings

  Explainability                      GATv2 attention weights

  Hardware                            Personal laptop with NVIDIA GPU;
                                      exact model and VRAM to be verified

  Expected duration                   14 days
  -----------------------------------------------------------------------

These decisions form the baseline. The following sections specify how to
verify and implement them.

------------------------------------------------------------------------

## 3. Datasets and Data Management

### 3.1 FineFS --- Primary Dataset

**Reference:** Fine-grained Figure Skating Dataset (FineFS), repository
indicated in the review:\
https://github.com/yanliji/FineFS-dataset

According to the documentation examined in the review, FineFS contains:

-   1,167 performances;
-   segmented technical sub-elements with time intervals;
-   GOE (*Grade of Execution*) scores associated with the elements;
-   17-joint 3D skeletons, distributed in `.npz` files;
-   a skeleton collection indicated in the review as approximately 828
    MB.

FineFS is the primary dataset because it associates, at least according
to the available description, segmented elements, skeletons, and
element-level quality evaluations.

**Mandatory checks before training:**

1.  Verify that the repository, download links, and files are still
    accessible.
2.  Inspect the contents of several `.npz` files without assuming array
    names or shapes a priori.
3.  Verify dimensions, units, reference frame, and coordinate axis
    conventions.
4.  Verify whether the coordinates are genuinely 3D and which estimation
    procedure produced them.
5.  Reconstruct the relationship between performance ID, time interval,
    technical element, skeleton, and GOE.
6.  Verify whether the GOE is associated with the individual sub-element
    or only at a more aggregated level.
7.  Check for missing values, undetected keypoints, duplicates, and empty
    segments.

Training must not begin until the correspondence between skeleton,
element, and annotation has been verified.

### 3.2 MMFS --- Supplementary Dataset

**Reference:** MMFS repository indicated in the review:\
https://github.com/dingyn-Reno/MMFS

MMFS is considered a possible additional source of skeletons and
annotations, including GOE and Base Value scores according to the
description reported in the review.

Integration is not automatic. Before merging the data, it is necessary
to compare:

-   convention and ordering of the 17 joints;
-   dimensionality and coordinate system;
-   units of measurement and axis conventions;
-   skeleton extraction method;
-   temporal segmentation of elements;
-   definition and scale of GOE;
-   athlete and performance identifiers;
-   possible overlaps between FineFS and MMFS.

If this information is incompatible or cannot be reconstructed, MMFS
must be kept separate or excluded from the MVP. Samples must not be
concatenated solely because both contain 17 keypoints.

### 3.3 Selection of Technical Elements

The MVP should focus on a few elements, preferably held poses or
elements with a recognizable and relatively stable phase.

Examples to evaluate include:

-   Camel Spin;
-   Sit Spin;
-   Layback Spin.

The final choice must depend on:

1.  number of available samples;
2.  skeleton quality and completeness;
3.  availability of GOE for the individual element;
4.  segmentation consistency;
5.  biomechanical comparability across executions.

The selection must be made through an exploratory analysis of the
dataset. It must not be assumed that all listed elements are present in
sufficient quantities.

### 3.4 Unified Data Schema

Each sample must be represented by a record with at least the following
logical fields:

  Field                   Description
  ----------------------- -------------------------------------------------
  `sample_id`             Unique sample identifier
  `dataset_id`            Source dataset
  `performance_id`        Performance identifier
  `athlete_id`            Athlete identifier, if available
  `element_label`         Technical element class
  `start_frame`           Segment start frame, if available
  `end_frame`             Segment end frame, if available
  `keypoints`             3D coordinates of the 17 joints
  `keypoint_confidence`   Per-joint confidence, if available
  `goe`                   GOE associated with the element, if available
  `source_metadata`       Original metadata required for traceability

The internal format must preserve the original metadata. Conversion into
the common format must not remove the ability to trace back to the
source sample.

### 3.5 Data Splits

The training/validation/test split must be performed **by performance
and, whenever possible, by athlete**, not by individual frame.

This reduces the risk that nearly identical frames from the same
execution appear in different splits.

All samples derived from the same execution, including those generated
via augmentation, must remain in the same split as the original sample.

Split percentages should be defined after measuring the effective sample
count per class. As a starting point, 70% training, 15% validation, and
15% test can be used, adjusting them if the dataset size does not allow
a stable partition.

------------------------------------------------------------------------

## 4. Temporal Pose Representation

The original design calls for comparing an average of a few frames to
represent a pose or a phase of the element.

For each selected segment:

1.  identify the temporal window corresponding to the pose or phase of
    interest;
2.  select a limited number of frames;
3.  verify keypoint quality in each frame;
4.  canonicalize each skeleton;
5.  aggregate the coordinates or features of valid frames.

The baseline is the mean of the canonicalized coordinates of the
selected frames:

\[ `\bar`{=tex}{P} = `\frac{1}{T}`{=tex}`\sum`{=tex}\_{t=1}\^{T}P_t \]

where (P_t `\in `{=tex}`\mathbb{R}`{=tex}\^{17 `\times 3`{=tex}}) is the
canonicalized skeleton at frame (t).

**Warning:** averaging coordinates is appropriate only when the frames
represent the same phase and are sufficiently aligned. Frames belonging
to different kinematic phases must not be averaged, because the result
might not correspond to a pose actually assumed by the athlete.

For the MVP, a temporal window consistent with the available
segmentation must be used. The exact window size and number of frames
are hyperparameters to be verified empirically.

------------------------------------------------------------------------

## 5. 3D Geometric Canonicalization

### 5.1 Motivation

Raw coordinates depend on translation, scale, and orientation of the
reference frame. Two biomechanically similar poses could therefore
appear different to the network.

Before graph construction, three transformations are applied:

1.  root-centering;
2.  scale normalization;
3.  yaw alignment.

These transformations reduce certain geometric differences, but do not
guarantee full viewpoint invariance. In particular, yaw alignment does
not automatically correct differences in pitch, roll, perspective, or 3D
reconstruction errors.

### 5.2 Root-centering

The root is defined as the midpoint of the hips, assuming the 17-joint
COCO convention and verifying the indices against the source file:

\[ p\_{`\text{pelvis}`{=tex}} = `\frac{p_{11}+p_{12}}{2}`{=tex} \]

For each joint:

\[ p'*i = p_i - p*{`\text{pelvis}`{=tex}} \]

This transformation removes the global translation of the skeleton.

### 5.3 Scale normalization

The midpoint of the shoulders is defined as:

\[ p\_{`\text{shoulders}`{=tex}} = `\frac{p_5+p_6}{2}`{=tex} \]

The reference scale is the torso length:

\[ s = \|p\_{`\text{shoulders}`{=tex}}-p\_{`\text{pelvis}`{=tex}}\|\_2
\]

The coordinates are normalized:

\[ p''\_i = `\frac{p'_i}{s}`{=tex} \]

If (s) is zero, too small, or invalid, the sample must be marked as
invalid or handled with an explicit strategy. One must never divide by
zero or silently replace missing values.

Normalization reduces differences in scale and body build, but may also
remove information about body proportions. This choice must be
documented as a model assumption.

### 5.4 Yaw alignment

The vector between the hips is used:

\[ v\_{`\text{hip}`{=tex}} = p\_{12}-p\_{11} \]

In the X-Y plane, compute:

\[ `\theta `{=tex}= `\operatorname{atan2}`{=tex}(v_y,v_x) \]

Apply an inverse rotation around the Z axis:

\[ R_z(-`\theta`{=tex})=
```{=tex}
\begin{bmatrix}
\cos(-\theta)&-\sin(-\theta)&0\\
\sin(-\theta)&\cos(-\theta)&0\\
0&0&1
\end{bmatrix}
```
\]

The canonicalized coordinates are:

\[ p'''\_i = R_z(-`\theta`{=tex})p''\_i \]

The implementation must verify that the X, Y, and Z axes of the datasets
have the meaning assumed by this formula. If the axis convention
differs, the rotation must be adapted.

### 5.5 Checks and Limitations

The pipeline must log:

-   torso length used for scaling;
-   estimated yaw angle;
-   presence of missing or invalid joints;
-   any discarded skeletons;
-   coordinate distributions before and after canonicalization.

Behavior in the case of a mirrored skeleton must also be verified.
Aligning the hip vector does not by itself resolve left/right
ambiguity. An automatic flip must not be applied without a verified rule
consistent with the labels.

Robustness across different viewpoints must be evaluated experimentally
on available data. It is not necessary to have the same execution filmed
from two cameras, but the absence of multi-view pairs prevents directly
demonstrating invariance to different viewpoints.

------------------------------------------------------------------------

## 6. Graph Construction

### 6.1 Topology

Each pose is represented as a graph:

\[ G=(V,E,X,E_f) \]

where:

-   (V): 17 nodes corresponding to the joints;
-   (E): anatomical connections of the skeleton;
-   (X): node features;
-   (E_f): edge features.

The topology must be fixed and consistent with the 17-keypoint standard
adopted by the datasets. The exact list of connections must be defined
once, verified against the HRNet/COCO convention used, and saved as a
project configuration.

Topologies that differ in node ordering or semantics must not be assumed
equivalent.

### 6.2 Node Features

Initial node features must describe the canonicalized geometry.
Baseline:

\[ x_i = \[x_i,y_i,z_i\] \]

Derived features may be added, such as bone direction vectors or
validity/confidence indicators, if available and verified.

For a bone between nodes (i) and (j):

\[ u\_{ij}=`\frac{p_j-p_i}{\|p_j-p_i\|_2+\epsilon}`{=tex} \]

Additional features must be introduced in a controlled manner, comparing
the baseline against the extended variant.

### 6.3 Edge Features

Edges must contain geometric information so that GATv2 does not consider
connectivity alone.

Candidate features:

-   normalized bone direction vector;
-   bone length after normalization;
-   cosine of the associated joint angle, when definable.

For three joints (i,j,k), the cosine of the angle at (j) is:

\[ `\cos`{=tex}(`\theta`{=tex}\_{ijk}) =
`\frac{(p_i-p_j)\cdot(p_k-p_j)}`{=tex}
{\|p_i-p_j\|\_2\|p_k-p_j\|\_2+`\epsilon`{=tex}} \]

The final representation of `edge_attr` must have constant dimensionality
for all edges and all samples.

**Implementation note:** if a feature is defined at the joint level
rather than per single edge, a deterministic rule must be established to
associate it with incident edges. This rule must be documented and kept
identical in training and inference.

### 6.4 PyTorch Geometric Format

Each graph must be serializable as a `torch_geometric.data.Data` object,
with at least:

-   `x`: node features;
-   `edge_index`: connectivity;
-   `edge_attr`: edge features;
-   `y` or separate metadata: annotations used by the sampler and
    evaluation;
-   sample and dataset identifiers preserved in metadata or an external
    index.

Serialization must allow reconstructing the original sample.

------------------------------------------------------------------------

## 7. Model Architecture

### 7.1 Siamese GATv2

The network consists of two branches with shared weights. Each branch
receives a graph and produces an embedding in the same latent space.

Diagram:

``` text
Graph A ──> Shared GATv2 Encoder ──> Embedding zA ──┐
                                                    ├─> Latent distance
Graph B ──> Shared GATv2 Encoder ──> Embedding zB ──┘
```

For triplet training:

``` text
Anchor   ──> Shared Encoder ──> zA
Positive ──> Shared Encoder ──> zP
Negative ──> Shared Encoder ──> zN
```

### 7.2 Choice of GATv2Conv

PyTorch Geometric's `GATv2Conv` is used instead of standard `GATConv`.

Design motivation: GATv2 enables a dynamic attention mechanism, in which
the weight assigned to neighbors depends more flexibly on the querying
node and the features involved.

The configuration must include `edge_dim` to allow the layer to use
edge features:

``` python
GATv2Conv(
    in_channels=...,
    out_channels=...,
    heads=...,
    edge_dim=...
)
```

### 7.3 Encoder

The encoder must include:

1.  one or more GATv2 layers;
2.  activation function;
3.  optional dropout;
4.  global pooling to obtain a graph-level representation;
5.  linear projection layer into the latent space.

Conceptual diagram:

``` text
Node features + Edge features
          |
        GATv2
          |
        GATv2
          |
  Global Mean Pooling
          |
   Linear projection
          |
      Embedding z
```

Initial dimensions to experiment with:

-   embedding: 32 or 64;
-   attention heads: 4 as initial configuration;
-   dropout: to be selected via validation;
-   number of layers: limited to reduce the risk of overfitting.

These values are initial hyperparameters, not already validated results.
The input and output dimensions of the layers must be consistent with
the number of heads and any head concatenation.

### 7.4 Shared Weights

The three inputs of the triplet must pass through the same encoder
instance. Independent encoders with separate parameters must not be
created.

### 7.5 Regularization

Given the potentially limited number of real samples:

-   keep the encoder compact;
-   use dropout;
-   apply early stopping on validation;
-   monitor the distance between embeddings;
-   avoid augmentation before splitting;
-   control overlap between triplets;
-   compare training and validation loss to detect overfitting.

------------------------------------------------------------------------

## 8. Training: Triplet Loss and Triplet Construction

### 8.1 Motivation

Precise penalty labels are not assumed to exist for every individual
pose. GOE, when associated with the individual element, is used to
construct quality relationships between samples.

Triplet Loss allows learning a metric without needing to know an exact
numerical penalty for each pose.

### 8.2 Triplet Definition

Each triplet contains:

-   **Anchor (A):** an execution of a given element;
-   **Positive (P):** an execution of the same element with comparable
    or higher quality;
-   **Negative (N):** an execution of the same element with lower
    quality.

Triplets must respect the same element class to prevent the model from
learning to distinguish different technical elements instead of
execution quality.

An initial rule, to be verified on the actual GOE distribution, is:

-   positives: samples with high GOE, e.g., GOE ≥ +2;
-   negatives: samples with low or non-positive GOE, e.g., GOE ≤ 0.

These thresholds are those proposed in the review and must be adapted to
the actual scale, distribution, and definition of GOE in the dataset.
They are not official per-joint penalty thresholds.

When possible, the Anchor should be chosen from high-quality samples and
the Positive from samples of similar quality. Selection must avoid using
the same execution as both Anchor and Positive.

### 8.3 Triplet Margin Loss

The standard loss is:

\[ L(A,P,N)= `\max`{=tex}`\left`{=tex}( \|f(A)-f(P)\|\_2 -
\|f(A)-f(N)\|\_2 + `\alpha`{=tex}, 0 `\right`{=tex}) \]

where (f) is the Siamese encoder and (`\alpha`{=tex}) is the margin.

The network must learn to maintain:

\[ d(A,P)+`\alpha `{=tex}\< d(A,N) \]

### 8.4 Adaptive Margin

A margin depending on the severity of the quality difference between
Anchor and Negative is planned.

Initial values proposed in the review:

  Severity of synthetic/annotated negative    Initial margin
  ----------------------------------------- ----------------
  Mild                                                   0.2
  Moderate                                               0.6
  Severe                                                 1.0

These values are experimental hyperparameters. They do not represent an
official conversion between GOE and latent distance.

**Important methodological point:** an adaptive margin requires a
reproducible rule that maps the Anchor-Negative pair to a difference
level. If GOE does not allow determining such a level, a simpler strategy
must be used: a fixed margin or quality tiers defined on the available
data.

### 8.5 Data Augmentation

Geometric augmentation is a supporting strategy, not a replacement for
real annotations.

Possible perturbations:

-   controlled variations of joint angles;
-   local perturbations of elbows, knees, or hips;
-   bounded Gaussian noise on coordinates;
-   small rotations and geometric variations compatible with the pose.

Every transformation must preserve anatomical plausibility and must not
accidentally change the element class.

Synthetic samples must carry metadata identifying the source sample, the
type of perturbation, and its intensity.

Official ISU penalties must not be automatically assigned to geometric
perturbations. Angular thresholds and penalty values mentioned in the
review must be verified against the applicable regulations and not
treated as certified labels without such verification.

Augmentation must be applied exclusively to samples in the training set.

### 8.6 Sampling Strategy

The sampler must:

1.  select an element class;
2.  select an Anchor;
3.  select a Positive of the same class and comparable quality;
4.  select a Negative of the same class but lower quality;
5.  assign the margin according to the configured rule;
6.  avoid samples belonging to different splits.

In a later phase, *semi-hard negative mining* can be introduced,
choosing negatives that are harder to distinguish. For the first MVP, it
is preferable to implement a simple, verifiable, and reproducible
sampler.

------------------------------------------------------------------------

## 9. Converting Distance into a Quality Indicator

The Euclidean distance between embeddings is:

\[ d(A,R)=\|f(A)-f(R)\|\_2 \]

where (A) is the athlete's sample and (R) is a high-quality reference of
the same element.

Distance is the primary output of the model. A smaller distance
indicates greater similarity in the learned space.

To present a more intuitive indicator, a monotonic transformation can be
applied, for example:

\[ q(d)=`\exp`{=tex}(-`\beta `{=tex}d) \]

where (`\beta`{=tex}) is a parameter to be calibrated on validation.

The result can be reported on a 0--100 scale:

\[ Q=100`\cdot `{=tex}q(d) \]

This value is an **internal similarity index**, not an official score, a
certified correctness percentage, or a direct estimate of GOE.

Calibration must be performed only on validation. The test set must
remain separate until final evaluation.

If multiple references are available for an element, the athlete can be
compared against a prototype obtained by aggregating the embeddings of
high-quality samples, or against multiple individual references. For the
MVP, it is sufficient to define a single strategy and keep it constant
during evaluation.

------------------------------------------------------------------------

## 10. Explainability

The explainability component uses the attention weights returned by
`GATv2Conv`.

With `return_attention_weights=True`, it is possible to retrieve the
attention coefficients associated with the edges.

Visualization objective:

-   render the skeleton on a 2D plane;
-   associate visual intensity with the connections;
-   highlight the connections to which the model assigns higher
    attention;
-   visually compare execution and reference.

The visualization must be interpreted with caution: a high attention
weight indicates that the layer assigns greater weight to a connection
in its computation, but does not by itself prove causality or direct
correspondence with a regulatory penalty.

For a readable visualization:

1.  aggregate weights across heads using an explicit rule, such as the
    mean;
2.  define a value normalization;
3.  map values onto the skeleton edges;
4.  display the element name and latent distance;
5.  maintain the same color scale across compared samples.

Requesting attention weights should be enabled during
inference/analysis, without unnecessarily slowing down training.

------------------------------------------------------------------------

## 11. Inference Pipeline and Demo

The demo must show operation on a few videos outside the training set.

Planned pipeline:

``` text
Video
  |
Pose estimation (MediaPipe or other selected estimator)
  |
Keypoint mapping to the canonical 17-node standard
  |
Keypoint quality check
  |
Temporal window selection
  |
3D canonicalization (if compatible 3D coordinates are available)
  |
Graph construction
  |
Siamese GATv2
  |
Embedding and distance from reference
  |
Similarity index + visualization
```

### 11.1 MediaPipe-HRNet Compatibility

The review proposes using MediaPipe for the demo, projecting its 33
landmarks onto the 17 joints of the COCO standard.

This operation requires an explicit and verified mapping for each joint.
Selecting indices without checking landmark semantics is not
sufficient.

Moreover, compatibility is not only topological:

-   HRNet and MediaPipe may produce coordinates in different reference
    frames;
-   3D coordinates may have different meanings and scales;
-   pose estimator bias may differ;
-   projecting from 33 to 17 points does not automatically make the
    distributions equivalent.

**Constraint:** the model trained on FineFS/MMFS skeletons must not
directly receive MediaPipe coordinates without a domain and
representation verification phase.

For the first MVP, it is acceptable to show the inference pipeline on a
sample coming from the same domain as the training data. The MediaPipe
demo is an extension subordinate to technical compatibility.

### 11.2 Alternative Pose Estimators

The review mentions MediaPipe as a practical solution for the demo, but
points out its limitations in complex sports contexts. It also cites
RTMPose and ViTPose as alternatives.

The final choice of pose estimator for external videos must depend on
the availability of compatible coordinates, observed quality, and
computational cost. Training a pose estimator is not required in the
MVP.

------------------------------------------------------------------------

## 12. Experimental Evaluation

Evaluation must measure both the ability to rank executions by quality
and the ability to satisfy triplet constraints.

### 12.1 Primary Metrics

**Triplet Accuracy**

Percentage of triplets for which:

\[ d(A,P)\<d(A,N) \]

This is the metric directly tied to the metric learning objective.

**Spearman Rank Correlation ((`\rho`{=tex}))**

Measures the monotonic correlation between predicted distance and
annotated quality. Since a smaller distance should correspond to higher
quality, the sign of the correlation must be handled consistently: one
can correlate distance with expected negative GOE or correlate (-d) with
GOE.

Correlation must be computed on independent samples with annotations
referring to the same element.

### 12.2 Additional Analyses

-   distance distribution across GOE tiers;
-   mean intra-group and inter-group distance;
-   training and validation loss curves;
-   performance per technical element;
-   count of valid/discarded samples;
-   sensitivity to noise and perturbations;
-   comparison between base features and extended features.

If the number of samples is small, results must be accompanied by the
sample count and interpreted as preliminary evidence, not as a robust
estimate of generalization.

### 12.3 Non-Neural Baseline

To justify using the GNN, it is necessary to compare it against at least
one geometric baseline.

Recommended baselines:

1.  Euclidean distance between canonicalized coordinates;
2.  distance between joint angles;
3.  optionally cosine similarity between feature vectors.

The GNN is justified if it learns a metric more consistent with quality
differences than these simple measures. This comparison is important
because nodes are already semantically aligned across graphs.

------------------------------------------------------------------------

## 13. Big Data and Computational Management

The main computational load is not necessarily GNN training, as 17-node
graphs are small. The most expensive part may be downloading, decoding,
and extracting poses from raw videos.

The chosen strategy is:

-   prioritize already-extracted skeletons in academic datasets;
-   perform preprocessing and canonicalization offline;
-   save preprocessed graphs to disk;
-   train the GNN on serialized graphs;
-   use raw videos only for a limited demo.

The project does not require introducing Spark, Kafka, or Hadoop, as no
specific course frameworks were mandated during discussion. The Big Data
component should be motivated through data pipeline management,
standardization, preprocessing, and the ability to scale extraction to
large video collections.

### 13.1 Hardware

The indicated device is an Asus Vivobook Pro 15 with an AMD Ryzen 6000
series processor and an NVIDIA GPU, but the exact GPU model and VRAM
have not been confirmed.

Before configuring training:

-   detect GPU model and VRAM;
-   verify compatible CUDA version;
-   install compatible versions of PyTorch and PyTorch Geometric;
-   run an end-to-end test on a few graphs.

Training must be designed to work on CPU or with small batches as well.
Planning must not rely on an assumed amount of VRAM.

------------------------------------------------------------------------

## 14. Proposed Software Structure

A possible repository organization:

``` text
figure-skating-aqa/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── graph.yaml
│   └── training.yaml
│
├── data/
│   ├── raw/
│   ├── intermediate/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_results_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   ├── harmonization.py
│   │   ├── segmentation.py
│   │   └── preprocessing.py
│   │
│   ├── graph/
│   │   ├── topology.py
│   │   ├── features.py
│   │   └── builder.py
│   │
│   ├── training/
│   │   ├── triplet_sampler.py
│   │   ├── augmentation.py
│   │   ├── losses.py
│   │   └── train.py
│   │
│   ├── models/
│   │   ├── encoder.py
│   │   └── siamese_gat.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   ├── predict.py
│   │   └── video_pipeline.py
│   │
│   └── visualization/
│       ├── skeleton_plot.py
│       └── attention_plot.py
│
├── checkpoints/
├── reports/
└── tests/
    ├── test_dataset.py
    ├── test_normalization.py
    ├── test_graph.py
    └── test_triplets.py
```

This structure is an organizational proposal. It can be simplified if
available time requires it, but data loading, preprocessing, graph
construction, model, and evaluation should remain separated.

------------------------------------------------------------------------

## 15. Technology Dependencies

Planned stack:

-   Python;
-   NumPy;
-   SciPy, if needed for geometric operations;
-   PyTorch;
-   PyTorch Geometric;
-   scikit-learn for utilities and metrics;
-   Matplotlib for visualizations;
-   OpenCV for optional video reading;
-   MediaPipe only for the demo, if compatible.

Versions must be pinned in `requirements.txt` after verifying
compatibility with the available Python, GPU, and CUDA versions.

------------------------------------------------------------------------

## 16. Tests and Acceptance Criteria

### 16.1 Data Pipeline Tests

-   A valid source file is loaded without metadata loss.
-   The number of joints is verified and consistent with the expected
    standard.
-   Coordinates have documented shape and dimensionality.
-   GOE and temporal segment are associated with the correct sample.
-   Invalid samples are identified and accounted for.

### 16.2 Canonicalization Tests

-   Root-centering brings the root to the origin within a numerical
    tolerance.
-   The reference torso length becomes unit length, except in invalid
    cases.
-   Yaw alignment brings the hip vector to the expected canonical
    orientation.
-   The function handles missing values and degenerate denominators.
-   The transformation does not alter joint ordering or identity.

### 16.3 Graph Tests

-   Each graph contains 17 nodes.
-   All graphs use the same topology.
-   `edge_index` respects node indices.
-   `edge_attr` has constant dimensionality and finite values.
-   The graph is compatible with PyTorch Geometric batching.

### 16.4 Training Tests

-   The three branches use the same parameters.
-   Triplets respect the element class.
-   Positive and Negative respect the configured quality rule.
-   The loss is finite and differentiable.
-   Training is reproducible given the same seed and configuration.
-   Training and validation are separated by performance/athlete
    according to the defined split.

### 16.5 MVP Success Criteria

The MVP is considered implemented when:

1.  a preprocessed and traceable dataset is available;
2.  at least one technical element has sufficient samples for training
    and preliminary evaluation;
3.  the pipeline produces canonical 17-node graphs;
4.  the Siamese GATv2 completes training without errors;
5.  Triplet Accuracy is above random chance on the test set;
6.  distance shows an interpretable relationship with GOE, if annotations
    are compatible;
7.  a comparison with a geometric baseline is available;
8.  at least one attention weight visualization is produced;
9.  limitations, splits, hyperparameters, and results are documented.

No arbitrary numerical threshold for Spearman or Triplet Accuracy is set
before analyzing the sample count and data distribution. Results must be
reported in full, even if negative.

------------------------------------------------------------------------

## 17. 14-Day Operational Roadmap

  -----------------------------------------------------------------------
  Days                    Activities              Deliverables
  ----------------------- ----------------------- -----------------------
  1--3                    FineFS download; `.npz` Exploratory report and
                          analysis; GOE,          dataset decision
                          segments, joints, and   
                          coordinates check; MMFS 
                          evaluation              

  4--6                    Element selection;      Preprocessed and tested
                          athlete/performance     PyG dataset
                          split;                  
                          canonicalization; graph 
                          construction and        
                          serialization; sampler  
                          and augmentation        

  7--10                   GATv2 encoder           Checkpoints and
                          implementation; shared  training report
                          Siamese; Triplet Loss;  
                          training and initial    
                          metrics                 

  11--14                  Test evaluation;        MVP, results, and
                          geometric baseline;     documentation
                          attention; limited      
                          demo; report and        
                          presentation            
  -----------------------------------------------------------------------

### Priorities in Case of Delays

1.  Complete FineFS and validate the data format.
2.  Use a single well-represented technical element.
3.  Implement canonicalization and graph construction.
4.  Train the Siamese GATv2 with a simple sampler.
5.  Compute metrics and compare against a baseline.
6.  Add MMFS only if integration is quick and verified.
7.  Add the MediaPipe demo only after verifying coordinate
    compatibility.
8.  Complete attention weight visualization.

The demo must not compromise model validation.

------------------------------------------------------------------------

## 18. Risks and Open Decisions

  -----------------------------------------------------------------------
  Risk / Open Point       Impact                  Action
  ----------------------- ----------------------- -----------------------
  FineFS coordinates not  High                    Inspect files and
  understood or not                               documentation before
  truly 3D in expected                            preprocessing
  format                                          

  GOE not associated with High                    Verify annotation
  individual element                              granularity; do not use
                                                  as local label if it is
                                                  not

  FineFS-MMFS             High                    Compare schema,
  incompatibility                                 coordinates, joints,
                                                  and annotations;
                                                  exclude MMFS if not
                                                  harmonizable

  Few samples per element High                    Reduce number of
                                                  elements and apply
                                                  controlled augmentation

  Imbalance across GOE    Medium/High             Analyze distributions
  tiers                                           and design balanced
                                                  sampler

  Overfitting             High                    Compact encoder,
                                                  athlete-level split,
                                                  dropout, early stopping

  HRNet/MediaPipe         High                    Do not run direct
  coordinate differences                          inference without
                                                  domain alignment
                                                  testing

  Left/right ambiguity    Medium                  Verify joint convention
                                                  and explicitly handle
                                                  any mirrored cases

  Incomplete viewpoint    Medium/High             Document limitations
  invariance                                      and evaluate on data
                                                  with available views

  Improper interpretation Medium                  Present weights as
  of attention                                    model indicators, not
                                                  causal explanations

  Uncalibrated            Medium                  Use similarity index
  distance-to-score                               and calibrate only on
  conversion                                      validation
  -----------------------------------------------------------------------

### Decisions to Finalize Before Training

-   Final technical elements and minimum acceptable sample count.
-   Exact 17-joint convention and edge list.
-   Actual names and shapes of FineFS arrays.
-   Reference frame and units of 3D coordinates.
-   Granularity and scale of GOE scores.
-   Temporal aggregation strategy.
-   Rule for reference selection.
-   Fixed or adaptive margin for Triplet Loss.
-   Similarity index calibration strategy.
-   Actual compatibility of the MediaPipe demo.

------------------------------------------------------------------------

## 19. References

References indicated in the project review:

-   FineFS Dataset: https://github.com/yanliji/FineFS-dataset
-   MMFS: https://github.com/dingyn-Reno/MMFS
-   PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
-   GATv2 paper: *How Attentive are Graph Attention Networks?* --- Brody
    et al., 2021.
-   FineGym, cited in the review as an alternative for gymnastics:
    https://sdolivia.github.io/FineGym/
-   MMAction2 skeleton data documentation:
    https://github.com/open-mmlab/mmaction2

FineGym and MMAction2 references are retained as methodological
alternatives/documentation and are not part of the primary dataset
chosen for the MVP.

------------------------------------------------------------------------

## 20. Final Architecture Summary

``` text
FineFS (.npz)                  MMFS (optional)
     |                              |
     +--------- verification -------+
                    |
             Data harmonization
                    |
         Element selection + GOE
                    |
          Athlete/event split
                    |
         Frame window selection
                    |
          Temporal aggregation
                    |
           3D canonicalization
      (root-centering, scale, yaw)
                    |
         Canonical 17-node graph
          /                   \
   Node features           Edge features
          \                   /
              Siamese GATv2
                    |
             Graph embedding
                    |
           Triplet Margin Loss
          (configurable margin)
                    |
           Athlete embedding
                    |
        Distance from reference
                    |
           Similarity index
                    |
       Metrics + explainability
```

**Guiding principle:** the GNN must learn a quality metric from
consistent graphs and verifiable annotations. Data quality, proper split
separation, and canonicalization validity are prerequisites of the
model, not secondary details.
