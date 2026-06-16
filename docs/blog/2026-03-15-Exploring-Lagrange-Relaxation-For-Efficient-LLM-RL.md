---
title:  "Exploring Lagrange Relaxation for Efficient LLM RL"
date:   2026-03-15
---

_This is an ongoing post on token-efficient LLM RL. Any suggestions for experiments or feedback are more than welcome!_

## TL;DR

I studied using Lagrangian relaxation in-depth in order to manage response length and format compliance in LLM RL without manual penalty tuning. 
Preliminary results show that, at least in the RLVR context, **most RL methods are not as effective as a traditional implementation of Lagrange relaxation** for controlling these auxiliary objectives while preserving task performance.

## Introduction

Recently I found out about Constrained Markov Decision Processes (CMDPs) (represented by [CPO](https://arxiv.org/abs/1705.10528), [RCPO](https://arxiv.org/abs/1805.11074), and interestingly, [TRPO](https://arxiv.org/abs/1502.05477) itself), which are a variant of the standard MDP framework where the agent must optimize some primary reward **while respecting constraints on secondary cost channels**. 
 
This problem formulation appears quite interesting to LLM RLVR, wherein we usually only care about some primary reward (e.g. task performance), but also have separate behaviors that we want to control (e.g. response length, format compliance), either because they improve training throughput, or they make training significantly easier (among many, MANY other potential reasons).

Thus, this blogpost is sort of an exploration of classical RLVR methods through the lens of constrained optimization.


## But First, Some Background...

### Group Relative Policy Optimization (GRPO)
In LLM RL, policy-gradient methods tend to be favored for their relative simplicity and ease of implementation (although [recent works](https://arxiv.org/abs/2602.19362) appear [to be revisiting this decision](https://openreview.net/forum?id=RduOiisl1S)). The most well known of these methods, [(Outcome Supervision) GRPO](https://arxiv.org/abs/2402.03300), attempts to optimize a contextual bandit objective of the form:
$$\max_{\theta} J(\theta) = \max_{\theta} \mathbb{E}_{q \sim Q, \{\tau_i\}_{i = 1}^G \sim \pi_\theta(\cdot|q)}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|} \frac{\pi_\theta(a_t^i|s_t^i)}{\pi_{\theta_{old}}(a_t^i|s_t^i) }\hat{A}(\tau_i, t)\right]$$
where $Q$ is some prompt distribution, $\tau_i$ represent individual responses to the prompt, and $\hat{A}(\tau_i, t) = \frac{r(\tau_i) - \mu_{\tau}}{\sigma_{\tau}}$ is the group-relative advantage. (Note: I omit KL here because it is not relevant to the formal discussion)

### Auxiliary Rewards in RLVR

As mentioned previously, reward functions in LLM RL are often made up of **primary** objectives (e.g. "Get Question Correct") and **secondary** objectives (e.g. "Don't produce malformed output"). This is a common practice in LLM RL as explained above (though [hardly](https://arxiv.org/abs/2007.03964) [unique](https://arxiv.org/abs/1705.10528) to it.)

Enough has been said about the **primary** objectives and how their apparent sparsity can still result in great empirical performance (how can I NOT link [DeepSeek-R1](https://arxiv.org/abs/2501.12948) when talking about this...), so let's focus instead on the **secondary** objectives, which can quite a bit more complex than we give them credit for.

#### Case Study: Format Compliance As A Secondary Objective

Most commonly, you will see  **format rewards** in mathematics or coding domains as a secondary objective. 

In these domains, responses are cheaply verifiable (via empirical or formal checks), given they come in a easily-parsed format, so it becomes much easier to give a "yes/no" reward signal in general.

[The](https://arxiv.org/abs/2402.03300) [most](https://arxiv.org/abs/2505.24864) [common](https://arxiv.org/abs/2501.12948) [approach](https://arxiv.org/abs/2504.13837) to designing formatting rewards is to **penalize ill-formatted responses with some negative constant reward**. 
 (while base models have gotten stronger at format following, [omit at your own risk](https://arxiv.org/pdf/2411.15124)...). This is a very simple way to encourage format compliance, but it conveniently downplays the non-triviality of **actually figuring out that constant amount to begin with**:
- Set it too low, and the model will never output format-compliant responses, making the task of actually giving it rewards difficult.
- Set it too high, and the model may learn to prioritize format compliance over task quality, leading to suboptimal performance on the primary objective.

Notice while we ask ourselves these questions, we also have **NO GOOD IDEA** of what "too low" or "too high" even means. It is doubtless that we know we can't set the format penalty to 0; that is a no-op. We also know the format penalty can't outscale the primary reward, otherwise the model will just learn to be format-compliant and ignore task quality. In between those two very obvious extremes, however, it becomes very hard to ascribe numeric thresholds and interpretations to the coefficients without just running a sweep and seeing what happens.

This is especially true when model behavior shifts during training, which can change the effective scale of the primary reward and thus the relative importance of the auxiliary reward.

#### Case Study: Response Length As A Secondary Objective

Of all the secondary objectives commonly adopted in LLM RL, perhaps the one which has captivated **industrial** research interest most recently is **response length**. We already know vanilla GRPO [tends to](https://arxiv.org/pdf/2501.12948) **generate longer and longer outputs**, and this is undesirable for a few reasons:
- **Higher Inference Cost**: More forward passes are needed to generate responses, increasing the cost of RLVR training
- **Lower Training Throughput**: Longer responses can lead to slower rollouts, which can bottleneck training speed and increase wall-clock time.
- **Blabbering**: At inference time, the model may produce unnecessarily long outputs when it is uncertain, which can degrade user experience.

<figure>
    <img src="../assets/Images/latte/r1-length-increase.png" alt="GRPO length increase" width="600"/>
    <figcaption>
    When training <a href="https://arxiv.org/abs/2402.03300">DeepSeek-R1-Zero</a>, DeepSeek noticed an appreciable increase in response length as GRPO training progresses
    </figcaption>
</figure>

[It turns out, there is a structural reason for this bias](https://arxiv.org/abs/2503.20783), and that is that **the GRPO objective systematically encourages wrong yet verbose** responses. Although I couldn't identify **WHY** the specific design choice was made [in the original GRPO paper](https://arxiv.org/abs/2402.03300), this phenomenon can be visualized with a very nice picture from [this paper]((https://arxiv.org/abs/2503.20783)):

![](../assets/Images/latte/grpo-bias.png)

The key intuition is that **in GRPO, all tokens "share" the final advantage signal equally**:
- If the advantage is negative, it becomes **less negative** when the response is longer, which can encourage the model to add more tokens to dilute the negative signal. 
- Conversely, if the advantage is positive, it becomes **more positive** when the response is shorter, which has its own set of problems (like poor-quality reasoning traces).

##### So... Shorter Is Better?

Not so fast.

While not exactly obvious, it is quite logical why shorter responses are not universally superior either:
- **Reward Hacking**: If we naively penalize length, the model can game it by outputting "<EOS>" all the time.
- **Suppressed Reasoning**: We know from [earlier](https://arxiv.org/abs/2201.11903) [research](https://arxiv.org/abs/2210.03629) can actually arise from contemplative **behaviors**, which in turn help task performance.

That changes the "response length" issue from a simple "shorter is better" problem into a **problem of tradeoffs**.

#### Challenges of Secondary Objectives in LLM RL

By considering both case studies above, we can summarize the main challenges of secondary objectives in LLM RL as follows:
1. **Difficulty of Manual Tuning**: It can be difficult to find the right scalar penalty for an auxiliary objective, and this can change during training as model behavior shifts.
    - Usually, people get around this with parameter sweeps and intuition. But is that necessarily the best way to do it? 
2. **Reward Misspecification**: It is very easy for unexpected consequences to arise from even seemingly simple reward formulations, such as overemphasizing format compliance at the cost of task performance, or encouraging laconic but low-quality responses.

## Constrained RL and Lagrange Relaxation

It turns out, however, that it is relatively easy to reframe these objectives as **soft constraints** (i.e. think of them as  tolerable up to a certain point):
- Formatting rewards: "As long as a sufficiently high percentage of responses are well-formatted, I can mark the LLM's responses correctly and let RL take the wheel from there."
    - Mathematically, $p(\text{format violation}) \le s_{\text{fmt}}$ for some setpoint $s_{\text{fmt}}$.
- Length rewards: "If I could control the average response length to be below some threshold, I could get a better grasp on my training budget and increase training throughput."
    - Mathematically, $\mathbb{E}[\text{response length}] \le s_{\text{len}}$ for some setpoint $s_{\text{len}}$.

See where I'm going with this? This turns our LLM RL problem into a **constrained optimization problem** (extremely loosely, a CMDP).

This is useful to us because [a common technique used to solve CMDPs](https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf) is indeed, Lagrangian relaxation. Given a problem that looks like this:

$$\max_{\theta} J(\theta)
\quad\text{s.t.}\quad
\mathbb{E}[c_k(\tau)] \le b_k,\;\; k=1,\dots,K,$$

Lagrangian relaxation helps re-state the problem as a **simpler dual problem** (see [Bertsekas](http://www.athenasc.com/nonlinbook.html) for more details):
$$\max_{\theta} J(\theta) - \sum_{k=1}^{K} \lambda_k \big(\mathbb{E}[c_k] - b_k\big) \quad \text{s.t.} \quad \lambda_k \ge 0.$$

wherein after each policy update, the multipliers $\lambda_k$ are updated by the subgradient method ([also displayed in RCPO](https://arxiv.org/abs/1805.11074)):
$$\lambda_k \leftarrow \left[ \lambda_k + \eta_k \big(\widehat{c}_k - b_k\big) \right]_+.$$

If you thought I did this just to wave some cool math at your face... guilty as charged? But more importantly for LLM RL,it means that we can change our line of questioning from "how do we tune arbitrary coefficients to achieve our desired behavior" to "what is our desired behavior, and how can our coefficients adapt to achieve it?"
- We have a mechanism to target some **goal** rather than some obtuse **penalty coefficient**, and
- we can **adaptively** update our coefficients to achieve that goal.

## Research Questions

The above background exposition gives rise to an interesting research question in LLM RL, especially given that [recent](https://arxiv.org/abs/2503.04697) [papers](https://arxiv.org/abs/2602.14468) have started to explore adaptive methods for controlling **response length** in particular:
1. How do classical constrained RL methods like Lagrangian relaxation perform in the context of LLM RLVR for controlling auxiliary objectives like format compliance and response length?
2. How do they compare to more recent methods that have been proposed for similar purposes, such as [LACONIC](https://arxiv.org/abs/2602.14468) and [L1](https://arxiv.org/abs/2503.04697)?

## Experiments

All experiments are math RLVR runs trained on `./data/train/deepscaler_40k` and evaluated on `./data/evaluation_suite`. The runs use the `qwen_math` prompt template, the math oracle, and the OAT training stack. Unless otherwise stated, the policy optimizer is Dr. GRPO.

### Training Setup

| Setting | Value |
| --- | --- |
| Hardware | 8x RTX 3090 |
| Training epochs | 3 |
| Training dataset | `./data/train/deepscaler_40k` |
| Evaluation suite | `./data/evaluation_suite` |
| Prompt template | `qwen_math` |
| Oracle | `math` reward oracle |
| Precision | FP16 |
| Base checkpoint | `Qwen/Qwen2.5-1.5B` |
| Math checkpoint | `Qwen/Qwen2.5-Math-1.5B` |

### Salient Hyperparameters

| Hyperparameter | Value |
| --- | --- |
| Optimizer family | PPO-style RLVR |
| Critic / advantage estimator | Dr. GRPO, except GRPO ablations where noted |
| Learning rate | `1e-6` |
| Learning-rate schedule | Constant |
| PPO epochs per update | `1` |
| KL coefficient / beta | `0` |
| Rollouts per prompt | `8` |
| Global batch size | `128` |
| Per-device train batch size | `2` |
| Prompt max length | `1024` tokens |
| Generation max length | `2048` tokens |
| Max model length | `3072` tokens |
| Rollout temperature / top-p | `1.0` / `1.0` |
| Evaluation temperature / top-p | `0.6` / `0.95` |
| Evaluation frequency | Every `16` training steps |
| Lagrange update rate | `0.02` |
| Format violation setpoint | `0.05` |
| Length setpoint | `0.25` of max generation length |
| Multiplier cap | `0.5` |

### Implementation Variants

| Method | Implementation | Auxiliary-control behavior |
| --- | --- | --- |
| Control | `train_control.sh`, `train_control_qwenmath.sh` | Dr. GRPO with fixed format penalty `0.1` and no active length penalty. Used to measure the unconstrained length drift under the same logging/evaluation setup. |
| Fixed penalties | `train_fixed_penalties.sh` | Manual scalar baseline with fixed format penalty `0.1` and fixed length penalty `0.1`; no multiplier updates. |
| Lagrange, no decay | `train_ablate_decay.sh` | Adaptive format and length multipliers initialized at `0.0`; update rate `0.02`; format violation setpoint `0.05`; length setpoint `0.25` of max generation length; multiplier cap `0.5`; no decay term. |
| LATTE / Lagrange with decay | `train_latte.sh`, `train_latte_qwenmath.sh` | Same adaptive format and length constraints as above, but with decay on both multipliers. Length cost is clamped to penalize only over-budget completions. |
| Length-only Lagrange | `train_len_penalty.sh` | Fixed format penalty `0.1`; adaptive length multiplier with decay, initialized at `0.0`, setpoint `0.25`, update rate `0.02`, cap `0.5`. |
| LACONIC | `train_laconic.sh`, `train_laconic_qwenmath.sh` | LACONIC learner with fixed format penalty `0.1`; only the length multiplier is adaptive. The reward penalizes positive length excess while leaving shorter completions unpenalized. |
| L1 Exact | `train_l1_exact.sh` | LCPO-style target-length prompting with reward `correctness - alpha * |target_length - response_length| - format_penalty`; `alpha=0.0003`, `delta=0.5`. |
| L1 Max | `train_l1_max.sh` | LCPO-style target-length prompting with sampled target lengths from 512 to 2048 tokens and clipped length bonus; `alpha=0.0003`, `delta=0.5`. |

### Experiment Matrix

| Group | Run | Model | Script | Purpose | Results |
| --- | --- | --- | --- | --- | --- |
| Base model | Control | `Qwen/Qwen2.5-1.5B` | `train_control.sh` | Establish Dr. GRPO baseline with fixed format penalty and no length control. | TODO |
| Base model | Fixed penalties | `Qwen/Qwen2.5-1.5B` | `train_fixed_penalties.sh` | Compare adaptive constraints against manually chosen constant format/length penalties. | TODO |
| Base model | Lagrange, no decay | `Qwen/Qwen2.5-1.5B` | `train_ablate_decay.sh` | Test plain projected multiplier updates for both format and length constraints. | TODO |
| Base model | LATTE | `Qwen/Qwen2.5-1.5B` | `train_latte.sh` | Main adaptive Lagrange run with decay on both format and length multipliers. | TODO |
| Base model | Length-only Lagrange | `Qwen/Qwen2.5-1.5B` | `train_len_penalty.sh` | Isolate the length constraint while keeping format handling fixed. | TODO |
| Base model | LACONIC | `Qwen/Qwen2.5-1.5B` | `train_laconic.sh` | Compare against a recent adaptive length-control method. | TODO |
| Base model | L1 Exact | `Qwen/Qwen2.5-1.5B` | `train_l1_exact.sh` | Compare against LCPO exact target-length matching. | TODO |
| Base model | L1 Max | `Qwen/Qwen2.5-1.5B` | `train_l1_max.sh` | Compare against LCPO max-style target-length control. | TODO |
| QwenMath model | Control | `Qwen/Qwen2.5-Math-1.5B` | `train_control_qwenmath.sh` | Check whether the same baseline behavior holds for a math-specialized checkpoint. | TODO |
| QwenMath model | LATTE | `Qwen/Qwen2.5-Math-1.5B` | `train_latte_qwenmath.sh` | Test whether adaptive Lagrange control transfers to the math-specialized checkpoint. | TODO |
| QwenMath model | LACONIC | `Qwen/Qwen2.5-Math-1.5B` | `train_laconic_qwenmath.sh` | Compare LATTE against LACONIC on the math-specialized checkpoint. | TODO |
