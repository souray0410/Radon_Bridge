# Retrieved reference excerpts — not the final attachments

Source: conversation 6a982e21-c098-83ed-970b-09a86c0e50ea, 统一三维重建概念.
Retrieved with read_thread on 2026-09-04. The final Householder files are only content-reference placeholders; attachments is empty. The available V1.0 text ends during section 18. Do not label this as the complete final specification. Later Householder correction supersedes the older Givens language.

# 1. Project Mission

The objective of this project is to develop **R&B (Radon Bridge)**:

> a general communication mechanism that allows neural representations of different intrinsic dimensionalities, architectures, modalities and tasks to exchange information through a common generalized Radon/projection domain.

The long-term scientific target is a Nature Biomedical Engineering-level study.

However, the project must be designed so that it can be **progressively expanded or safely narrowed** according to experimental evidence, available data and project time.

The core engineering principle is therefore:

\[
\boxed{
\text{Build the method generally from the beginning;
validate the scientific claims progressively.}
}
\]

Do NOT initially hard-code R&B for only:

- 2D–3D;
- CFP–OCT;
- eye–heart;
- one particular task;
- one particular backbone;
- one particular network depth.

The mathematical and software implementation should be dimension-general from the beginning.

The experiments will be enabled progressively.

---


# 2. Existing Codebase Constraint

Before implementing R&B, inspect the latest version of the **MHD Project** provided by the user.

R&B must be implemented consistently with the design philosophy and abstractions already used in that project.

In particular:

1. do not replace the existing framework with an independent conventional PyTorch architecture simply because that is easier;
2. understand how the MHD Project represents networks and computation;
3. preserve its hypergraph-based representation of forward and backward computation;
4. integrate R&B into that representation rather than creating an unrelated parallel framework;
5. reuse existing tensor, node, edge, module, training and configuration abstractions wherever appropriate;
6. keep the R&B implementation modular enough that it can be inserted into arbitrary feature nodes represented in the MHD framework.

The exact integration design should be decided only after inspecting the actual MHD Project code.

The present document defines the **scientific and algorithmic requirements**, not the final software API.

---


# 4. Central R&B Principle

R&B does NOT permanently convert all representations into the same dimensionality.

Instead, each participating neural feature temporarily performs:

\[
\boxed{
\text{Native Space}
\rightarrow
\text{Projection Space}
\rightarrow
\text{Communication}
\rightarrow
\text{Native Space}
}
\]

The backbone remains responsible for nonlinear representation learning.

R&B is responsible only for structured communication.

The complete conceptual sequence is:

\[
\boxed{
\textbf{Project}
\rightarrow
\textbf{Reshape}
\rightarrow
\textbf{Handoff}
\rightarrow
\textbf{Interact}
\rightarrow
\textbf{Expand}
\rightarrow
\textbf{Backproject}
\rightarrow
\textbf{Residual Return}
}
\]

The default interaction mechanism should remain **linear**.

This is intentional.

R&B should demonstrate that useful cross-dimensional communication can emerge from a structured common geometry without requiring a powerful nonlinear fusion network inside the bridge itself.

---


# 5. Native Feature Nodes

An R&B participant is a neural feature node:

\[
F_i.
\]

It may originate from:

- any network;
- any network depth;
- any modality;
- any supported intrinsic dimensionality;
- any downstream task.

Examples may eventually include features originating from:

- ECG;
- CFP;
- OCT;
- cardiac cine imaging;
- other biomedical sources.

Two participating nodes do NOT need to:

- have the same number of channels;
- have the same spatial size;
- have the same dimensionality;
- belong to the same architecture;
- occur at the same layer index;
- perform the same downstream task;
- have the same output space.

The only requirement is that the node can be mapped through the appropriate generalized projection operator.

---


# 6. Communication Groups

One R&B operation acts on a set of participating nodes:

\[
\mathcal G
=
\{F_1,F_2,\ldots,F_K\}.
\]

This is called a **communication group**.

R&B is inherently multi-input and multi-output.

The input and output node identities must be identical:

\[
\boxed{
\{F_1,F_2,\ldots,F_K\}
\rightarrow
\{F_1',F_2',\ldots,F_K'\}.
}
\]

A node enters the communication group, receives information influenced by the other participants, and returns to its own original position.

Do NOT implement arbitrary routing such as:

\[
F_A\rightarrow\text{R\&B}\rightarrow F_D
\]

when \(F_D\) was not a participant.

The conceptual rule is:

> The same participants enter and leave the meeting; only their information has changed.

---


# 13. Mesh Control

Let:

\[
M_j^0
\]

be the full/reference number of samples on angular axis \(j\).

Then:

\[
M_j
=
\max
\left(
1,
\operatorname{round}
(
\upsilon_M M_j^0
)
\right).
\]

Important:

Reducing:

\[
\upsilon_M
\]

must NOT shrink:

\[
[0,\pi)
\]

to a smaller angular range.

Instead, it reduces the density of the Orientation Mesh while maintaining complete angular coverage.

Thus:

\[
\upsilon_M
\]

controls the projection-orientation budget.

---


# 14. Span Control

For each participant \(i\), determine a reference/full-support projection resolution:

\[
S_i^0.
\]

The precise definition must remain configurable until real feature dimensions and physical spacings are inspected.

A candidate reference rule is based on the maximum spatial diagonal/support required to cover all projections.

For one communication group:

\[
S_{\mathcal G}^0
=
\max_{i\in\mathcal G}
S_i^0.
\]

Then:

\[
S_{\mathcal G}
=
\max
\left(
1,
\operatorname{round}
(
\upsilon_S S_{\mathcal G}^0
)
\right).
\]

All participants map their complete signed-distance support onto this common number of Span locations.

Reducing:

\[
\upsilon_S
\]

reduces the discretization density.

It must NOT crop the signed-distance support.

Important scientific caution:

Do not describe the full diagonal/support-based configuration as mathematically “lossless” unless that is formally established.

Use language such as:

- full-support reference;
- native-resolution reference;
- complete projection support.

Exact sampling/reconstruction guarantees depend on interpolation, angular sampling and discrete implementation.

---


# 16. Handoff Control

For participant \(i\):

\[
P_i\times S
\]

is linearly compressed to:

\[
H_i\times S.
\]

Define:

\[
H_i
=
\max
\left(
1,
\operatorname{round}
(
\upsilon_H P_i
)
\right).
\]

Important:

Do NOT force:

\[
H_1=H_2=\cdots=H_K.
\]

Using the same proportional control:

\[
\upsilon_H
\]

is preferred over forcing every participant to the same absolute Handoff width.

Therefore:

\[
H_1,\ldots,H_K
\]

may differ.

This preserves a comparable proportional communication budget across representations of very different sizes.

---


# 17. Handoff Compression Location

The default method must perform Handoff compression:

\[
\boxed{
\text{after Radon projection and after projection-channel reshape}.
}
\]

Do NOT define the primary Handoff operation as native backbone channel compression before Radon.

Reason:

- pre-Radon compression changes the native representation before entering the proposed common geometry;
- post-Radon Handoff compression acts directly on the communication representation;
- M, S and H therefore retain clearly separated scientific meanings.

If memory constraints later require pre-Radon compression, treat it as:

- an implementation optimization;
- or a controlled ablation;

not as the default definition of Handoff.

---


# Stage 3 — M–S–H Representation

实现：

\[
\boxed{
M=\text{Orientation Mesh}
}
\]

\[
\boxed{
S=\text{Projection Span}
}
\]

\[
\boxed{
H=\text{Communication Handoff}
}
\]

以及控制向量：

\[
\boxed{
\boldsymbol{\Upsilon}
=
(\upsilon_M,\upsilon_S,\upsilon_H)
}
\]

验证三者真正彼此独立。

---

## Mesh Test

改变：

\[
\upsilon_M
\]

时：

- angular support 仍为 \([0,\pi)\)；
- orientation 数量改变；
- S 不应因此被重新定义；
- Handoff 比例规则不应改变。

---

## Span Test

改变：

\[
\upsilon_S
\]

时：

- signed-distance support 不被裁剪；
- 仅改变离散位置数量；
- angular mesh 不变；
- Handoff rule 不变。

---

## Handoff Test

改变：

\[
\upsilon_H
\]

时：

- native feature 不变；
- Radon projection 本身不变；
- Mesh 不变；
- Span 不变；
- 仅 reshape 后进入 communication 的 projection-channel capacity 改变。

### Stage 3 Gate

只有当：

\[
M,\;S,\;H
\]

三个控制轴能够分别独立改变时，才进入 interaction implementation。

---


# Stage 6 — Complete R&B Unit Synthetic Validation

到这里才第一次把完整链路连起来：

\[
F_i
\rightarrow
\text{Project}
\rightarrow
\text{Reshape}
\rightarrow
\text{Handoff}
\rightarrow
\text{Interact}
\rightarrow
\text{Expand}
\rightarrow
\text{Backproject}
\rightarrow
F_i'.
\]

使用 synthetic tensors 测试：

### Pairwise

\[
2D\leftrightarrow3D
\]

### Same-dimensional

\[
2D\leftrightarrow2D
\]

以及：

\[
3D\leftrightarrow3D.
\]

### Multi-node

例如：

\[
1D+2D+3D
\]

以及：

\[
1D+2D+3D+4D.
\]

### Heterogeneous shape

参与节点 channel 和 native spatial sizes 故意不同。

### Multiple R&B units

验证：

\[
\text{Backbone}
\rightarrow
R\&B_1
\rightarrow
\text{Backbone}
\rightarrow
R\&B_2
\]

能够正常 forward/backward。

---


# Stage 8 — First Scientific Gate

首先训练 independent baseline。

然后加入最简单的 R&B configuration。

建议首先测试：

\[
1\text{ R\&B}
\]

和：

\[
2\text{ R\&B stages}.
\]

不要一开始做复杂 topology search。

比较：

\[
\text{Independent}
\]

vs.

\[
\text{Independent + R\&B}.
\]

如果主要任务没有稳定改善：

停止扩展实验。

检查：

- projection correctness；
- S alignment；
- Handoff compression；
- interaction initialization；
- optimization；
- communication stage；
- task compatibility。

只有确认基本效果成立后，再进入下一阶段。

---


# Stage 15 — Training Regimes

至少支持三种训练方式。

## A. Pretrained + Frozen Backbone

先训练/加载原模型。

冻结 backbone。

只训练 R&B。

科学目的：

\[
\boxed{
\text{Does R\&B alone extract useful complementary information?}
}
\]

---

## B. Pretrained + Joint Fine-Tuning

插入 R&B 后一起 fine-tune。

这预计是主要性能结果。

---

## C. End-to-End From Scratch

所有网络和 R&B 同时随机初始化训练。

科学目的：

> R&B 是否依赖 pretrained representations？

不要把 training recipe 本身变成新的研究贡献。

---


## Later presentation statement

“所以它不是把几个网络彻底融合成一个网络，而更像是几个原本独立的网络，在中间几个阶段临时进入一个共同的 projection space 交流一下，然后各自继续自己的任务。”

## Later Householder revision

The final-file announcement explicitly changes Givens to Householder: H = I - 2vv^T, H^T = H, H^-1 = H, H^2 = I. Two final files are referenced but their bodies were not returned by read_thread.

