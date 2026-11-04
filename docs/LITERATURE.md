# Literature and novelty check (done before any code, 2026-11-05)

Every reference below was checked to exist (arXiv API or the venue page) on 2026-11-05.

## What is already known

**In-context learning of function classes.**
- Garg, Tsipras, Liang, Valiant (2022), *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes*, arXiv:2208.01066. Transformers trained on random linear functions in-context match least squares.
- Akyürek, Schuurmans, Andreas, Ma, Zhou (2022), *What learning algorithm is in-context learning? Investigations with linear models*, arXiv:2211.15661. Behaviour matches ridge/OLS depending on depth and noise. They probe for intermediate quantities (the moment X^T y and the weight vector) in **static i.i.d. linear regression**.
- von Oswald et al. (2023), *Transformers learn in-context by gradient descent*, arXiv:2212.07677 (ICML 2023). One linear self-attention layer can implement one GD step; trained linear models match the construction.
- Ahn, Cheng, Daneshmand, Sra (2023), arXiv:2306.00297: preconditioned GD. Zhang, Frei, Bartlett (2023), arXiv:2306.09927: training dynamics of linear attention for linear regression. Vladymyrov, von Oswald, Sandler, Ge (2024), arXiv:2402.14180.
- Fu, Chen, Jia, Sharan (2023), arXiv:2310.17086: trained transformers look like iterative Newton, not GD; layer-wise predictions are matched to Newton iterates.
- Bai, Chen, Wang, Xiong, Mei (2023), *Transformers as Statisticians*, arXiv:2306.04637 (NeurIPS 2023). Constructions for ridge, Lasso, GLMs and in-context algorithm selection; least squares with O(log) depth.
- Raventós, Paul, Chen, Ganguli (2023), arXiv:2306.15063. With enough task diversity, ICL departs from the Bayes posterior over the *training* tasks and approaches ridge with the true prior.
- Müller et al. (2022), *Transformers Can Do Bayesian Inference*, arXiv:2112.10510. Training on prior samples yields approximate posterior predictive distributions.
- Olsson et al. (2022), *In-context Learning and Induction Heads*, arXiv:2209.11895.

**In-context learning of dynamical systems.**
- Goel & Bartlett (2023), *Can a Transformer Represent a Kalman Filter?*, arXiv:2312.06937 (L4DC 2024). Constructive: a softmax transformer approximates the KF **for a known system**.
- Balim, Du, Oymak, Ozay (2023), *Can Transformers Learn Optimal Filtering for Unknown Systems?*, arXiv:2308.08536. Behavioural: a meta-trained transformer approaches KF performance on unseen linear systems.
- Akram & Vikalo (2024; TMLR 03/2026), *Transformers as Implicit State Estimators*, arXiv:2410.16546. Behavioural match to KF / EKF / PF; graceful degradation when the transition matrix is withheld. Their future-work list explicitly includes studying the internal representations that support in-context filtering.
- Cole, Zhao, Lu, Zhang (2025), arXiv:2502.08136 (NeurIPS 2025). Approximation bounds: O(log)-depth linear transformers reach least-squares rates for in-context LDS learning; single-layer lower bound.
- Vlădărean, Zhang, Sra (ICLR 2026), *On learning linear dynamical systems in context with attention layers*. The optimal single linear-attention layer is one GD step on the autoregression objective (window 1); a PCG connection for larger windows.
- Forgione, Pura, Piga (2023), *From system models to class models: An in-context learning paradigm*, arXiv:2308.13380 (IEEE L-CSS). Meta-models for one-step prediction and simulation of whole system classes. Follow-ups: Piga, Pura, Forgione (2023), arXiv:2312.04083 (adaptation); Busetto et al. (2023), arXiv:2312.04509 (in-context state estimators).
- Busetto, Breschi, Forgione, Piga, Formentin (2024), *One controller to rule them all*, arXiv:2411.06482. A transformer used as a **contextual controller** for a system class (model-reference). Behavioural only.

**Adaptation from history and in-context RL.**
- Kumar, Fu, Pathak, Malik (2021), *RMA: Rapid Motor Adaptation for Legged Robots*, arXiv:2107.04034. A history encoder regresses a latent of privileged physical parameters.
- Probing RMA-style latents gives mixed results: Chamachot (2026), arXiv:2603.21268, finds weak factor structure. Zhang, Zhang, Mueller (2026), arXiv:2608.23887, decode control latents back to physical parameters with a learned flow model.
- Laskin et al. (2022), *In-context RL with Algorithm Distillation*, arXiv:2210.14215. Duan et al. (2016), RL², arXiv:1611.02779. Zintgraf et al. (2019), VariBAD, arXiv:1910.08348: Bayes-adaptive exploration emerges in meta-RL.
- Liang & Lai (2026), arXiv:2605.05755: constructions for in-context SARSA and actor-critic in **tabular** MDPs. Trained weights recover the construction.

**Adaptive control background.**
- Mania, Tu, Recht (2019), arXiv:1902.07826: certainty equivalence is efficient for LQR.
- Simchowitz & Foster (2020), arXiv:2001.09576: naive exploration (dithering) is optimal for online LQR.
- Closed-loop identifiability and the dual effect: Feldbaum (1960); Åström & Wittenmark, *Adaptive Control* (1995); Ljung, *System Identification* (1999).

**Interpretability tools.** Alain & Bengio (2016), arXiv:1610.01644 (linear probes). Zhang & Nanda (2023), arXiv:2309.16042, and Heimersheim & Nanda (2024), arXiv:2404.15255 (activation patching).

## Is the core question already answered?

Partly. **"Does a transformer behave like the Kalman / least-squares estimator on in-context linear dynamical systems?"** is answered (Balim et al.; Akram & Vikalo), and so is **"can it represent one?"** (Goel & Bartlett; Cole et al.; Bai et al.). Re-doing either would not be a contribution.

What is **not** answered (searched 2026-11-05, nothing found):
1. **Mechanism in the dynamical / closed-loop setting.** Nobody has probed or causally intervened on a transformer doing in-context system identification of a dynamical system. In particular, nobody has asked whether it stores *sufficient statistics*, a *parameter estimate*, or *the true parameter*, in which layer, and whether that representation is causal. Akyürek et al. probe static i.i.d. regression only. Akram & Vikalo list it as future work.
2. **What an in-context *controller* computes.** Contextual controllers exist (Busetto et al. 2024), but nobody has asked whether a transformer trained only on control cost implicitly identifies the system, or whether it solves the **dual-control** problem: closed-loop data under exploration-free feedback cannot identify the plant, so a good adaptive controller must excite it. Meta-RL work shows exploration can emerge, but not with a quantitative test against the closed-loop identifiability theory.
3. **Physical parameters in contact-rich MuJoCo systems.** Here the optimal estimator is unknown. RMA-style probing studies supervise the latent with the parameters. We ask what an *unsupervised* in-context predictor encodes and whether steering it along a probed "mass" or "friction" direction changes its physics in the right direction.

## Sharpened question (what changed and why)

Original: *"what algorithm does a transformer learn when it adapts to unknown physics in-context?"*

Sharpened: ***Does an in-context transformer adapt to unknown dynamics by (i) accumulating the estimator's sufficient statistics and then solving for a Bayes-optimal estimate, and (ii) when trained as a controller, does it add the excitation that closed-loop identifiability theory says is necessary? And are the internal quantities causal?***

Why: the behavioural half of the original question is settled. The mechanistic and control halves are open, and we can test them with exact Bayes-optimal references in the linear case and with interventions everywhere.

Theory contribution, chosen to avoid duplicating Goel & Bartlett / Cole et al. / Bai et al.:
- an exact Bayes-risk expression and context-length law for in-context prediction of linear systems with a Gaussian prior, giving a parameter-free target curve (n·p·σ²/t asymptotically);
- a closed-loop non-identifiability lower bound: under any exploration-free linear feedback, the posterior along the unexcited directions never contracts, whatever the context length;
- a two-layer construction that puts the *exact* sufficient statistics in the residual stream, which motivates the "statistics first, solve later" probing hypothesis.
