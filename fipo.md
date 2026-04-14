# FIPO: Eliciting Deep Reasoning with Future-KLInfluenced Policy Optimization

Qwen Pilot Team, Alibaba Group ∗

Project Page

GitHub

HuggingFace

ModelScope

# Abstract

We present Future-KL Influenced Policy Optimization (FIPO), a reinforce-ment learning algorithm designed to overcome reasoning bottlenecks in largelanguage models. While GRPO style training scales effectively, it typicallyrelies on outcome-based rewards (ORM) that distribute a global advantageuniformly across every token in a trajectory. We argue that this coarse-grainedcredit assignment imposes a performance ceiling by failing to distinguishcritical logical pivots from trivial tokens. FIPO addresses this by incorporatingdiscounted future-KL divergence into the policy update, creating a denseadvantage formulation that re-weights tokens based on their influence on sub-sequent trajectory behavior. Empirically, FIPO enables models to break throughthe length stagnation seen in standard baselines. Evaluated on Qwen2.5-32B,FIPO extends the average chain-of-thought length from roughly 4,000 to over10,000 tokens and increases AIME 2024 Pass@1 accuracy from $5 0 . 0 \%$ to a peak of${ \bf 5 8 . 0 \% }$ (converging at approximately $5 6 . 0 \%$ ). This outperforms both DeepSeek-R1-Zero-Math-32B $\mathrm { \sim 4 7 . 0 \% }$ ) and o1-mini $( \sim 5 6 . 0 \%$ ). Our results suggest thatestablishing dense advantage formulations is a vital path for evolving ORM-based algorithms to unlock the full reasoning potential of base models. Weopen-source our training system, built on the verl framework.

# 1 Introduction

Test-time scaling strategies such as those employed in OpenAI’s o-series (Jaech et al., 2024), Gemini series(Comanici et al., 2025), and DeepSeek’s R-series (Guo et al., 2025) mark a fundamental shift in how largelanguage models carry out reasoning. By allocating greater computational resources at inference time,these approaches support longer chain-of-thought and more deliberate reasoning, leading to substantialgains on demanding tasks such as competitive mathematics and coding. Much of this progress stemsfrom large-scale reinforcement learning with verifiable rewards (RLVR) (Guo et al., 2025; Team et al.,2025a; Yang et al., 2025; Team et al., 2025b; Zeng et al., 2025), which fine-tunes a model’s generation policyusing feedback from task-specific verifiers, thereby eliciting and amplifying its reasoning capabilities.However, since the specific algorithms and training recipes remain largely undisclosed, it is still unclearhow reinforcement learning serves as the primary catalyst to unlock potential reasoning depth, effectivelyeliciting the emergence of long chain-of-thought behaviors from base models that initially exhibit nosuch tendencies.

In parallel, the open-source community has devoted substantial effort to reproducing and scaling similaralgorithms in more transparent settings (Qin et al., 2024; Huang et al., 2024; Liu et al., 2025; Hu et al., 2025;Yu et al., 2025). Among these efforts, DAPO (Yu et al., 2025) provides a promising large-scale reproductionof GRPO-style training applied to clean base models. However, we argue that the inherent relianceon outcome-based rewards within the GRPO framework introduces a significant structural constraint.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/649960a08d78ce8bbf19a340162be165df442579eda8008c93cd0d0d30d62972.jpg)



Figure 1: FIPO vs. Baselines Performance Comparison on AIME2024. FIPO demonstrates that pureRL training alone is sufficient to not only outperform other pure RL baselines (the reproduced DAPOand Deepseek-R1-Zero-32B), but also surpass o1-mini. This performance gain is accompanied by thegeneration of significantly longer responses on average.


Because rewards are only binary-verifiable at the trajectory end, the standard formulation distributesa uniform advantage to every token. This results in a completely coarse-grained credit assignmentwhere the algorithm treats critical reasoning steps and trivial tokens with equal weight. Specifically,we observe that reasoning trajectories produced by such baselines tend to plateau at intermediate lengths.We contend that this limitation imposes a lower performance ceiling on standard GRPO: because theuniform reward cannot highlight the specific tokens that drive correct logic, the model is unable toconverge to the complex, extended reasoning paths needed for difficult tasks. While this limitation hasled recent works (Hu et al., 2025; Yue et al., 2025; Fan et al., 2025) to revert to the PPO framework forgranular advantage estimation, we contend that such density is achievable without the complexity of acritic model.

We introduce Future-KL Influenced Policy Optimization (FIPO). FIPO modifies the policy update byincorporating the Future-KL divergence, which re-weights the advantage of current tokens based onthe cumulative behaviors of their subsequent trajectories. To maintain training stability, this objectiveis coupled with influence weight clipping and filtering mechanism. We evaluate this approach onQwen2.5-32B-Base, a model with no prior exposure to long-CoT synthetic data, utilizing the publiclyreleased training dataset from DAPO (Yu et al., 2025) to ensure a strictly controlled comparison. Asshown in Figure 1, FIPO breaks the performance ceiling of standard baselines; while DAPO achieves$5 0 . 0 \%$ (Pass@1) on AIME 2024, FIPO enables a progressive lengthening of reasoning chains, wherethe model steadily scales from a baseline of 4,000 tokens to a deep-reasoning regime of over 10,000tokens. This consistent expansion pushes accuracy to a peak of $5 8 . 0 \% ,$ a result on par with recentPPO-based counterparts.These findings demonstrate that establishing a dense advantage formulationeffectively bridges the gap between GRPO efficiency and PPO performance, unlocking deep reasoningcapabilities that otherwise remain untapped under uniform reward schemes.

Our implementation is built upon the verl framework (Sheng et al., 2025) and the DAPO codebase. By

fully releasing the complete training code and configuration recipes, we aim to reveal valuable insightsinto large-scale reinforcement learning for LLMs that benefit the broader research community.

# 2 Related Work

Reinforcement Learning for LLMs. Reinforcement learning (RL) serves as a cornerstone of the post-training pipeline for large language models. While foundational efforts primarily utilized ReinforcementLearning from Human Feedback (RLHF) to align model behavior with human preferences (Stiennonet al., 2020; Ouyang et al., 2022), recent advancements have shifted focus toward enhancing reasoningcapabilities through RL. Notable examples include the OpenAI o-series (Jaech et al., 2024), whichpioneered this reasoning-centric approach, and DeepSeek-R1 (Guo et al., 2025), which introduced acomprehensive RLVR (Lambert et al., 2024) framework for developing reasoning models via the GRPOalgorithm (Shao et al., 2024). These breakthroughs have further inspired a wave of industry-leadingsubsequent works, such as Kimi (Team et al., 2025a), Qwen3 (Yang et al., 2025), and Gemini 2.5 (Comaniciet al., 2025).

Large-scale open-source RL recipes. Parallel to the proprietary advancements in reasoning models, theopen-source community has made significant strides in democratizing large-scale RL training. These ef-forts aim to bridge the gap between high-level algorithmic concepts and practical, stable implementationsthat can scale efficiently, while providing continuous improvements to the training pipeline. Notably,GSPO (Zheng et al., 2025), BAPO (Xi et al., 2025), SAPO (Gao et al., 2025), and OR1 (He et al., 2025)primarily develop their RL algorithms on models that have already developed long-CoT capabilities.Other works devote significant effort to incentivizing complex reasoning abilities starting from a cleanerbase model, specifically Qwen2.5-32B-Base. Among these efforts, Open-Reasoner-Zero (Hu et al., 2025),VC-PPO(Yuan et al., 2025), VAPO (Yue et al., 2025), and T-PPO (Fan et al., 2025) build their algorithmsupon the PPO framework (Schulman et al., 2017), whereas DAPO (Yu et al., 2025) is developed as amodification of GRPO.

To ensure a rigorous evaluation, we adopt Qwen2.5-32B-Base as our backbone and use DAPO as ourprimary baseline. While Open-Reasoner-Zero reverts to PPO to avoid the sparse advantage signalsin vanilla GRPO, we address this challenge by refining the GRPO framework directly. Notably, sinceOpen-Reasoner-Zero operates without auxiliary value models, its performance ultimately falls shortof DAPO. In contrast, other methods like VC-PPO, VAPO and T-PPO rely heavily on value modelsthat are pre-trained by models already supervised fine-tuned (SFT) with Long-CoT data. We contendthat this methodology introduces an external knowledge prior through the value model, creating apotential confounding factor in the evaluation. This makes it difficult to discern whether the performancegains stem from the policy optimization algorithm itself or are simply inherited from the pre-trainedvalue model. By eschewing the need for a value model and starting from a vanilla base model, FIPOachieves performance comparable to, and in some cases superior to, these pre-trained value-model-based approaches. This demonstrates that establishing a dense advantage formulation is a promisingdirection for evolving ORM-based GRPO algorithms to unlock the inherent reasoning potential ofbase models.

# 3 Preliminary

In this section, we review the policy optimization frameworks central to our work: PPO and its value-network-free variants, GRPO and DAPO. Throughout this paper, let T denote the total length of atrajectory and t denote the index of the current step within that trajectory. In the GRPO setting, for eachquestion prompt $q ,$ we sample G trajectories, yielding outputs denoted by o.

# 3.1 Proximal Policy Optimization

Proximal Policy Optimization (PPO) (Schulman et al., 2017) introduces a clipped surrogate objective forpolicy optimization. By constraining policy updates to the proximity of the old policy through a clippingmechanism, PPO stabilizes training and improves sample efficiency. Specifically, PPO maximizes:

$$
\mathcal {J} _ {\mathrm {P P O}} (\theta) = \mathbb {E} _ {(q, o) \sim \mathcal {D}, o \sim \pi_ {\theta_ {\mathrm {o l d}}} (\cdot | q)} \left[ \min  \left(r _ {t} (\theta) \hat {A} _ {t}, \operatorname {c l i p} (r _ {t} (\theta), 1 - \epsilon , 1 + \epsilon) \hat {A} _ {t}\right) \right].
$$

Here, $\begin{array} { r } { r _ { t } ( \theta ) = \frac { \pi _ { \theta } \left( o _ { t } | q , o _ { < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( o _ { t } | q , o _ { < t } \right) } } \end{array}$ denotes the token-level probability ratio at step t, $\hat { A } _ { t }$ is the advantage estimatedvia a learned value function, and $\epsilon$ is the clipping coefficient. Crucially, standard PPO implementationscompute the advantage $\hat { A } _ { t }$ using Generalized Advantage Estimation (GAE) (Schulman et al., 2015).This results in distinct, token-specific advantage signals, enabling the model to perform temporal creditassignment. This stands in contrast to simplified formulations that derive advantages solely from thefinal outcome, effectively broadcasting a uniform signal to all tokens within a trajectory. By leveragingGAE, PPO provides dense supervision at every step, allowing it to differentiate between critical and lessinfluential actions along the generation process.

# 3.2 Group Relative Policy Optimization

Group Relative Policy Optimization (GRPO) (Shao et al., 2024) circumvents the computational burdenof a value network by estimating advantages through group-based sampling. For a given query $q$ (andground truth a), a set of outputs $\{ o _ { i } \} _ { i = 1 } ^ { G }$ is sampled from the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ . The sequence-level advantagefor the $i .$ -th sample is standardized as:

$$
\hat {A} _ {i} = \frac {R _ {i} - \mu}{\sigma}, \quad \text {w i t h} R _ {i} = \mathbb {I} (\operatorname {V e r i f y} (o _ {i}, a)), \tag {1}
$$

where $\mu$ and $\sigma$ denote the empirical mean and standard deviation, respectively, of the rewards within thesampled group. Similar to PPO, GRPO adopts a clipped objective but adds a per-token KL penalty termdirectly to the loss:

$$
\begin{array}{l} \mathcal {J} _ {\mathrm {G R P O}} (\theta) = \mathbb {E} _ {q \sim \mathcal {D}, \left\{o _ {i} \right\} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\text {o l d}}} (\cdot | q)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{\left| o _ {i} \right|} \sum_ {t = 1} ^ {| o _ {i} |} \left(\min  \left(\rho_ {i, t} (\theta) \hat {A} _ {i}, \operatorname {c l i p} \left(\rho_ {i, t} (\theta), 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {i}\right) \right. \right. \tag {2} \\ \left. \left. - \beta D _ {\mathrm {K L}} (\pi_ {\theta} | | \pi_ {\mathrm {r e f}})\right) \right]. \\ \end{array}
$$

Here, $\begin{array} { r } { \rho _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( o _ { i , t } | q , o _ { i , < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( o _ { i , t } | q , o _ { i , < t } \right) } } \end{array}$ represents the probability ratio. By design, the computed scalar $\hat { A } _ { i }$ isold broadcast across the entire sequence; specifically, for every token $t ,$ the advantage is set identically as$\hat { A } _ { i , t } = \hat { A } _ { i }$ . Unlike PPO, where Generalized Advantage Estimation (GAE) provides a distinct signal foreach token, GRPO assigns uniform credit to every step in the trajectory, regardless of its individualcontribution to the final outcome.

# 3.3 Decoupled Clip and Dynamic Sampling Policy Optimization

Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) (Yu et al., 2025) extends theGRPO framework by eliminating the explicit KL penalty. Instead, it employs asymmetric clipping withinthe interval $\big ( 1 - \epsilon _ { \mathrm { l o w } } , 1 + \epsilon _ { \mathrm { h i g h } } \big )$ to amplify updates for advantageous actions, effectively mitigatingthe entropy collapse commonly observed with GRPO. Furthermore, DAPO implements a token-levelpolicy gradient loss to sustain healthy optimization dynamics in the context of long Chain-of-ThoughtRL training. Furthermore, DAPO enforces a dynamic sampling mechanism that guarantees a mix ofpositive and negative samples within each group $\{ o _ { i } \} _ { i = 1 } ^ { G }$ . This mechanism ensures effective updates withnon-trivial gradients during optimization. We adopt DAPO as the primary baseline for this work.

# 3.4 Findings on Directions of Policy Update and Fine-grained Token Analysis

In our previous work, Meng et al. (2025) provides a systematic analysis on how RL rewrites the basemodel. We found that in over $9 8 \%$ of generation steps, the output distribution is identical. RL onlyintervenes at highly sparse, critical tokens to keep the model on track. Additionally, Huang et al. (2025)argue that Standard metrics (like KL divergence) fail to locate these sparse changes. By tracking thesigned log-probability difference, we can precisely map the “direction" of optimization, and even boostinference accuracy just by amplifying these key tokens, with zero extra training. These insights leadto a clear conclusion: not all tokens contribute equally to the reasoning process. However, while theinstantaneous log-probability difference indicates the direction of optimization, it serves merely as aprimitive, localized signal. The key to eliciting more effective reasoning then lies in discovering how toleverage this raw $\Delta \log p$ to formulate a much more accurate measurement of a token’s true downstreamimpact, thereby enabling us to automatically locate and reinforce these critical junctions during RLtraining.

# 4 FIPO

In this section, we introduce the core framework of FutureKL-Induced Policy Optimization (FIPO). Webegin by discussing the probability shift, the fundamental building block of our objective. Next, we detailthe formulation of Future-KL. Finally, we illustrate how our method implements a “soft decay window”strategy by focusing on the local “future context”. This mechanism naturally prioritizes proximal signalsover distant ones, limiting the effective horizon to the most relevant subsequent tokens.

# 4.1 Probability Shift: ∆ log p

Our method is grounded in our recent investigations into the dynamics of Large Language Models(LLMs) during reinforcement learning. Specifically, our previous work on RLVR updates (Huang et al.,2025) demonstrates that the magnitude and direction of the probability shift, $\Delta \log p ,$ , serve as robustindicators of improved reasoning. Building upon this, our fine-grained analysis of distributional shifts(Meng et al., 2025) further reveals that this generation process is often driven by a few “sparse but critical”tokens that disproportionately influence the subsequent chain of thought. Inspired by these insights,we identify the token-level probability shift as the atomic unit for our credit assignment mechanism.Formally, we define the probability shift at time step t as the log-space difference between the currentpolicy and the old policy:

$$
\Delta \log p _ {t} = \log \pi_ {\theta} \left(o _ {t} \mid q, o _ {<   t}\right) - \log \pi_ {\theta_ {\text {o l d}}} \left(o _ {t} \mid q, o _ {<   t}\right). \tag {3}
$$

This term serves as a differential signal capturing the instantaneous policy drift:

• Positive Shift $( \Delta \log p _ { t } > 0 )$ : Indicates that the current policy has increased the likelihood oftoken $o _ { t }$ relative to the old policy. This typically suggests that the training objective is reinforcingthis specific reasoning step.

• Negative Shift $\Delta \log p _ { t } < 0 )$ : Implies that the policy is suppressing the generation of $o _ { t } ,$ signalingthat the updated model is actively down-weighting this specific token relative to the referencepolicy.

Unlike traditional KL penalties, which treat this drift primarily as a regularization cost to be minimized,we interpret $\Delta \log p _ { t }$ as a directional signal of behavioral adjustment, thereby explicitly coupling theoptimization objective to the generative dynamics. However, relying solely on this instantaneous shift isinsufficient, as it fails to capture the long-term consequences of a decision. This limitation motivates ourproposed Future-KL mechanism, which re-weights the current token by aggregating the distributionalshifts of its future trajectory.

# 4.2 Future-KL Estimation

While $\Delta \log p _ { t }$ captures the local distributional shift, reasoning is inherently a sequential process wherethe true significance of this token depends on the trajectory it initiates. To capture this causal influence,we define Future-KL as the cumulative signed probability shift from the current step t to the end of thesequence T:

$$
\text {F u t u r e K L} _ {t} = \sum_ {k = t} ^ {T} \Delta \log p _ {k}. \tag {4}
$$

This summation is mathematically equivalent to the log-likelihood ratio of the joint probability distribu-tions for the subsequent sequence $o _ { t : T }$ . It can thus be interpreted as a sample-based estimate of the KLdivergence restricted to the future horizon, measuring the cumulative deviation of the current policyfrom the reference policy for the remainder of the trajectory. We therefore term this metric Future-KL.Functionally, Future $\mathrm { \mathbb { K L } } _ { t }$ serves as a forward-looking metric that quantifies the cumulative shift in policydistribution regarding the future trajectory. A positive value $( \mathrm { F u t u r e K L } _ { t } > 0 )$ ) indicates that the updatedpolicy has overall reinforced the entire subsequent trajectory initiated by token $o _ { t } ,$ , suggesting that $o _ { t }$acts as a stable anchor for the subsequent reasoning chain. In contrast, a negative value (FutureKL $\iota _ { t } < 0$ )implies that the policy is collectively suppressing the future tokens following $o _ { t } ,$ , signaling that thetrajectory stemming from this point is becoming less favored during the optimization process.

However, in practice, such formulation tends to exacerbate the variance arising from distributional shifts.Since FutureKLt acts as a weighting coefficient for the advantage function (as detailed in subsequentsections), excessive deviations in future logits (e.g., due to training-inference inconsistency) can dispro-portionately inflate the scale. This renders the optimization overly sensitive to noisy tokens rather thanthe intrinsic quality of the reasoning chain. Empirically, we observe that in the absence of safety mecha-nisms, training runs are prone to severe instability. As shown in Figure 2, this collapse is distinctivelyaccompanied by a sharp spike in the “low-clip fraction” metric, which tracks the frequency of samplestriggering the Dual-Clip threshold (a hard clip ratio on negative samples) (Ye et al., 2020). Such highimportance ratios on negative samples signify a critical misalignment: the model assigns high probabilityto an action that is effectively harmful. In our experiments, this spike (at approximately Step 70) alignswith a surge in the gradient norm and Policy $\mathrm { { K L ^ { 1 } } }$ , indicating a substantial shift in policy distribution,alongside an immediate drop in response length. This synchronization indicates that without regulation,the accumulated negative signals from FutureKLt can reach some extreme values that destabilize thetraining process.

Motivated by these observations, we refine the FutureKL computation by explicitly masking tokensthat exceed the Dual-Clip threshold. Since these tokens represent ’harmful’ actions whose gradients arealready clipped (via the clipped policy objective), allowing their excessively high importance ratios topropagate into the recursive sum introduces severe variance. By zeroing out the future accumulation forthese specific outliers, we remove the primary source of instability. The refined objective is defined as:

$$
\text {F u t u r e K L} _ {t} = \sum_ {k = t} ^ {T} M _ {k} \cdot \Delta \log p _ {k}, \quad M _ {k} = \mathbb {I} \left(\frac {\pi_ {\theta} \left(o _ {k} \mid o _ {<   t}\right)}{\pi_ {\mathrm {o l d}} \left(o _ {k} \mid o _ {<   t}\right)} \leq c\right). \tag {5}
$$

Here, $M _ { k }$ acts as a binary filter that evaluates to 1 only if the importance ratio remains within the Dual-Clipthreshold $c$ (typically $c \geq 1 0 ^ { }$ ), and 0 otherwise. This ensures that tokens triggering the hard constraintsare effectively excluded from the FutureKL computation, preventing gradient explosion without alteringthe trajectory’s valid signals.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/7c07961eddf9065e43c0c0c20d6d0fef2699ac94f318ff153c46b0ca096fb7e5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/063bbae6f8022569011fc405e213aa08db19774bf89aebe7e8d854a818ac36e9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/c42831d2cf07d7a9e4b4f5e97a2b4f21fe61d3ac1078261daa0c841f3f657049.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/5eb0732426eaf3e12d0981550a1bd230690d0ceea6f456b7089f7834dc01961a.jpg)



Figure 2: Training instability with vanilla FutureKL. Analysis of the unstable run observed around Step70. (a) A sharp spike in the low-clip fraction (indicating a drastic shift in the policy distribution driven bynegative samples) triggers (b) a sudden divergence in Policy KL. (c) an immediate explosion in gradientnorm and These internal instabilities collectively precipitate (d) a catastrophic collapse in response length,confirming that unregulated negative signals destabilize the optimization.


# 4.2.1 Soft Decay Window

Beyond the stability constraints, we also address the inherent uncertainty of long-horizon generation. Thecausal dependency between the current action $o _ { t }$ and future tokens $o _ { k }$ naturally diminishes as the timehorizon $k - t$ increases. Immediate successors are directly conditioned on the current choice, whereasdistant tokens are subject to accumulating stochasticity and become less predictable. To model thisdiminishing influence, we introduce a discount factor $\gamma \in ( 0 , 1 ]$ . Incorporating this decay into the maskedobjective yields the final formulation used in our experiments:

$$
\text {F u t u r e K L} _ {t} = \sum_ {k = t} ^ {T} M _ {k} \cdot \gamma^ {k - t} \cdot \Delta \log p _ {k}. \tag {6}
$$

We parameterize the decay rate as $\gamma = 2 ^ { - \frac { 1 } { \tau } }$ , where $\tau$ is a hyperparameter controlling the effectivehorizon (or “half-life”) of the future supervision. This formulation ensures that the credit assignmentconcentrates on the immediate reasoning chain, assigning lower weights to distant, highly uncertaintokens. Functionally, $\tau$ defines the aperture of this soft decay window. Unlike a hard truncation thatabruptly discards information beyond a fixed step, this exponential formulation creates a continuoussliding window where $\tau$ represents the distance at which the future signal’s influence attenuates by half.This mechanism allows the model to prioritize local coherence within the window $\tau _ { \cdot }$ , while smoothlyfiltering out the noise from the distant future without introducing boundary artifacts.

# 4.2.2 FutureKL Re-weighted Advantage with Clipping

Finally, we integrate the soft decay window and masking mechanisms into the policy optimizationobjective. We propose to modulate the standard advantage estimate $\hat { A } _ { t }$ using a future influence weight $f _ { t }$The modified advantage ${ \tilde { A } } _ { t }$ is defined as:

$$
f _ {t} = \operatorname {c l i p} \left(\exp \left(\text {F u t u r e K L} _ {t}\right), 1 - \epsilon_ {f _ {\text {l o w}}}, 1 + \epsilon_ {f _ {\text {h i g h}}}\right), \quad \tilde {A} _ {t} = \hat {A} _ {t} \cdot f _ {t}. \tag {7}
$$

This formulation introduces two key operations:

1. Exponential Mapping: We transform the accumulated scalar signal from log-space to a multi-plicative domain. Mathematically, the unclipped term represents a decay-weighted product oflikelihood ratios, which acts as an importance weight reflecting the policy’s effective preferencefor the generated future.

2. Influence Weight Clipping: We constrain the multiplicative coefficient $f _ { t }$ to the interval [1 −$\epsilon _ { f _ { l o w } } , 1 + \epsilon _ { f _ { h i g h } } ]$ . This operation serves strictly to bound the magnitude of the advantage modu-lation, preventing the exponential term from introducing excessive variance into the gradientestimate. By capping the weight, we ensure that the future trajectory modulates the updatesignal within a controlled range, avoiding numerical instability caused by extreme accumulatedlog-probability shifts.

Functionally, this modulation scales the magnitude of the policy update based on the reinforcementor suppression of the generated future. When the updated policy reinforces the subsequent trajectory(i.e., FutureKLt > 0), the weighting term $f _ { t } > 1$ magnifies the gradient signal. Consequently, positiveadvantages are boosted to encourage the current token as a stable anchor, while negative advantages incurharsher penalties to strictly correct errors initiating this path. Conversely, when the policy suppresses thefuture trajectory (i.e., Future $\mathrm { K L } _ { t } < 0 \}$ ), the term $f _ { t } < 1$ attenuates the update. This attenuation effectivelyreduces the reward signal for locally harmful tokens that happen to be in a successful sequence andsoftens the penalty for good tokens trapped in a failing one. In practice, to ensure training stability andprevent over-penalization, we reset $f _ { t } = 1$ for tokens associated with negative advantages $( \hat { A } _ { t } < 0 )$ thatexhibit excessively large importance ratios.

# 4.3 Target Loss

Adopting the token-level formulation from DAPO (Yu et al., 2025), we maximize the following FIPOobjective:

$$
J _ {\mathrm {F I P O}} (\theta) = \mathbb {E} _ {(q, a) \sim \mathcal {D}, \{o _ {i} \} \sim \pi_ {\theta_ {\mathrm {o l d}}}} \left[ \frac {1}{\sum_ {i = 1} ^ {G} | o _ {i} |} \sum_ {i = 1} ^ {G} \sum_ {t = 1} ^ {| o _ {i} |} \min  \left(r _ {i, t} f _ {i, t} \hat {A} _ {i, t}, \operatorname {c l i p} \left(r _ {i, t}, 1 - \epsilon , 1 + \epsilon\right) f _ {i, t} \hat {A} _ {i, t}\right) \right]. \tag {8}
$$

Here, G represents the number of sampled outputs per query, and $\begin{array} { r } { r _ { i , t } = \frac { \pi _ { \theta } ( a _ { i , t } | s _ { i , t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { i , t } | s _ { i , t } ) } } \end{array}$ denotes the impor-tance ratio between the current and old policies. The term $\hat { A } _ { i , t }$ refers to the group relative advantage,while $f _ { i , t }$ serves as the Future-KL importance weight introduced previously.

# 5 Experiment

# 5.1 Experiment Settings

In this work, we adopt the training settings of DAPO (Yu et al., 2025), specifically focusing on mathemati-cal reasoning tasks to ensure a strictly controlled comparison. We utilize the VeRL framework (Shenget al., 2025) for both training and baseline reproduction. We maintain optimization settings consistentwith DAPO, and trained on the public-released DAPO-17K dataset. Each training batch consists of512 prompts with 16 responses sampled per prompt, yielding a total of 8,192 training samples. In thestandard DAPO configuration, updates are performed with a mini-batch size of 512 samples (32 prompts),resulting in 16 gradient updates per training iteration. However, our empirical findings suggest that alarger mini-batch size improves training stability. Consequently, we adopted a mini-batch size of 1,024samples (64 prompts), resulting in 8 gradient updates per iteration. A more detailed discussion regarding

the impact of this increased minibatch size is provided in Appendix Sec. E. For the Future-KL compu-tation, we set the effective horizon of the decay rate $\tau$ to 32. Specific to the training of 32B model, theFuture-KL weight is clipped within [1, 1.2]; this effectively amplifies the reward for tokens associatedwith successful reasoning trajectories while imposing a more stringent penalty for those leading toincorrect outcomes. Both FIPO and DAPO share a maximum response length of 20,480 tokens, with anoverlong penalty applied to trajectories exceeding 16,384 tokens. Detailed hyperparameter configurationsfor both the baseline and FIPO are provided in Appendix Sec. A.

For evaluation, we adopt AIME 2024 as our primary validation benchmark, supplemented by AIME 2025,to ensure a rigorous and comprehensive comparison with the DAPO baseline. To maintain results stabilityand account for variance in chain-of-thought generation, we follow the DAPO protocol by repeating theevaluation 32 times and reporting the Pass@1 (averaged over 32 samples). Inference hyperparameters areconsistently set to a temperature of 1.0 and a top- $p$ of 0.7.

# 5.2 Main Result

Table 1 presents the quantitative evaluation on the AIME 2024 and AIME 2025 benchmarks. FIPO achievesa systematic improvement of roughly $6 . 0 \%$ in Pass@1 (Avg@32) over the DAPO baseline across bothdatasets. We prioritize this metric as the most robust indicator of reasoning reliability. While we alsoobserve gains in consistency, the improvement in coverage (Pass@32) is more modest, particularly onAIME 2025. We attribute this to the inherent challenge of expanding the absolute problem-solving scopeof large models through reinforcement learning alone. Without external knowledge augmentation ortool integration, RL is primarily constrained to refining how the model navigates its existing internalknowledge. Consequently, while FIPO significantly enhances the model’s ability to reliably solveproblems within its latent capacity (driving up Avg@32), shifting the boundary of solvable problems(Pass@32) remains non-trivial.


Table 1: Comparison of reasoning performance on AIME benchmarks. All results are reported aspercentages $( \% )$ . We report the average Pass $@ 1$ across 32 samples $( \mathrm { A v g } @ \mathcal { B } 3 2 )$ , the majority vote (Cons@32),and the probability of at least one correct answer (Pass@32). To align with prior baseline reporting andreduce sensitivity to digit-level generation variance, final values are rounded to the nearest integer.


<table><tr><td rowspan="2">Method</td><td colspan="3">AIME 2024</td><td colspan="3">AIME 2025</td></tr><tr><td>Avg@32</td><td>Cons@32</td><td>Pass@32</td><td>Avg@32</td><td>Cons@32</td><td>Pass@32</td></tr><tr><td>DAPO (Baseline)</td><td>50.0%</td><td>60.0%</td><td>80.0%</td><td>38.0%</td><td>47.0%</td><td>63.0%</td></tr><tr><td>FIPO (Ours)</td><td>56.0%</td><td>73.0%</td><td>83.0%</td><td>43.0%</td><td>50.0%</td><td>67.0%</td></tr></table>

# 6 Analysis

Beyond the aggregate metrics, we observe several distinct phenomena that we believe underpin theseperformance gains. By dissecting the training dynamics and inference behaviors, we identify three criticaldrivers of FIPO’s effectiveness: the emergence of length-based scaling in reasoning chains, the distinctpositive learning signal captured by the response length weighted mean advantage formulation, and thesignificantly improved stability of the optimization process compared to standard baselines.

# 6.1 The scaling of length and performance

A central observation in FIPO’s training is that performance gains are deeply coupled with a contin-uous expansion of response length. As training progresses, we observe a significant surge in tokencounts that scales alongside model accuracy. As illustrated in Figure 3, the response length of DAPOgradually enters a stagnation phase after an initial increase, plateauing at an average of approximately

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/20888da6869ec087c153e5e295640e39cfe8059155c9acfb980b78fe9b3a6d4d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/686a18b1c8a129f50af565b8cbd9715a017cfaea3a1e4c152b3a1b46e045890b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/448b7062ebc12deeb021d66929633dd8f6f29c4d5d1f6e94d5ad9309b80c1ff3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/7d611a16733927b3fd5f8621064da38cda6040624c894cdf20bdea94422ae938.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/d7ce4ed31feeb1b0ac0625deeeb8b85a99d95901a5a5000a350eed362c3256f3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/64cc2caa45741c92256e88a270f630a0db9c2cbd80d48f688a1f51ca9372ea92.jpg)



Figure 3: Dynamics of response length and performance scaling during training. Subplots (a)-(e) showthe evolution of response length metrics (Min, Q25, Mean, Median, Q75) over global training steps.Compared to the DAPO baseline, FIPO significantly increases response length, effectively eliciting moreextensive Chain-of-Thought reasoning. Subplot (f) demonstrates that this increased length correlatesstrongly with improved accuracy, suggesting that longer CoT is key to breaking performance barriers.


4,000 tokens. In contrast, FIPO exhibits remarkable scaling resilience. This scaling process unfoldsthrough distinct evolutionary phases (visualized by the colored regions in Figure 3), marking a transitionfrom an initial rapid exploration to a sustained period of deep reasoning. Notably, although an overlongpenalty is maintained to constrain redundancy, FIPO successfully guides the model to elicit extensiveChain-of-Thought (CoT) reasoning. Qualitative analysis provided in Appendix D reveals that this lengthexpansion is driven by the gradual emergence of self-reflection behaviors; the model increasingly utilizesthe expanded sequence length to re-evaluate its intermediate steps and explore multiple methodologies toverify its conclusions. Interestingly, this spontaneous emergence of systematic self-verification aligns withthe inference-time scaling behaviors observed in advanced reasoning models (e.g., the OpenAI o-seriesand DeepSeek-R1). This suggests that FIPO effectively triggers inference-time reasoning, prioritizinganalytical depth to unlock higher performance.

Further examination of the training dynamics reveals that this surge in length is not driven by isolatedoutliers but represents a comprehensive distributional migration. As shown in Figure 3(a)–(e), alllength-related percentiles, ranging from the Minimum and Q25 to the Median and Q75, exhibit a synchro-nized and stable upward shift under FIPO training. Specifically, across these training phases, the mediantoken count climbs steadily from an initial 200 to over 10,000. Such a migration across the entire distribu-tion demonstrates that FIPO facilitates a fundamental shift in the model’s underlying problem-solvingstrategy: the model transitions from direct response patterns to systematic, self-verifying reasoningprocesses. Crucially, we find that this collective shift toward longer reasoning chains is what unlocks theperformance breakthroughs observed in our experiments. As illustrated in Figure 3(f), there is a strongpositive correlation between model accuracy and response length across all identified stages. Whilethe correlation slopes (denoted as $w$ ) vary slightly between phases, the trajectory remains consistentlypositive. While the DAPO baseline’s performance reaches a bottleneck as its length plateaus, FIPO’sability to continuously unlock additional “thinking space” allows the model to navigate increasinglycomplex logical dependencies. This confirms that FIPO successfully converts increased sequence length

into genuine reasoning depth, enabling the model to surpass the performance ceilings of standardbaselines on high-difficulty reasoning tasks.

# 6.2 The dynamics of advantage and sustained reasoning growth

We further investigate the training dynamics by comparing the evolution of rewards and advantages. Asshown in Figure 4(a), the baseline (DAPO) consistently maintains a higher mean training reward thanFIPO. However, we argue this disparity is a numerical artifact of the reward formulation rather thanan indicator of superior performance. Because the reward function incorporates an overlong penalty,FIPO’s construction of elaborate reasoning chains inevitably leads to higher penalties, thus suppressingits average raw reward. Conversely, the baseline’s higher reward is driven by its tendency to generateshorter responses. While this strategy maximizes immediate reward by minimizing penalties, it suggestsa convergence to a local optimum within a restricted search space.

This hypothesis is further corroborated by the rapid escalation in the number of sampled batches forDAPO, as shown in Figure 4(b). This trend indicates that the model is overfitting the training set,increasingly generating non-discriminative samples (i.e., batches that are uniformly correct or incorrect)which yield negligible gradient information. Consequently, the algorithm is forced to sample moreaggressively to harvest sufficient effective data for optimization. In contrast, FIPO actively traverses amore expansive search space, prioritizing the structural depth required for challenging reasoning tasksover the mere avoidance of penalties.

This difference becomes even more pronounced when shifting from raw rewards to the dynamic incen-tives provided by advantages. As observed in Figure 4(c), DAPO exhibits a declining trend in responselength weighted mean relative advantage2 throughout training. This implies that the length of positivesamples is increasingly dominated by that of negative samples, resulting in a diminishing incentive toextend derivations; since increased length no longer yields more positive relative advantages, the modeleventually hits a plateau in reasoning growth. In stark contrast, FIPO demonstrates a consistent upwardtrajectory. This indicates that the positive samples are evolving to be significantly more substantive thantheir negative counterparts. This dynamic fosters a sustained growth trajectory: as the generation oflonger, valid reasoning chains yields increasingly positive advantages, facilitated by the steady rise inrewards, it preserves the model’s momentum to pursue even more extensive and rigorous reasoningpaths.

# 6.3 Smooth Policy Drift, Exploration and Gradient Update

To further characterize the training process, we examine the evolution of policy behavior and optimizationstability. As shown in Figure 5(a), FIPO exhibits a steady and structured increase in Policy KL divergence.This represents a progressive policy shift, where the model consistently moves away from its previouspolicy state to navigate toward a more specialized reasoning regime. This trend is qualitatively consistentwith our rollout observations: the length of self-reflection segments increases incrementally rather thanabruptly, reflecting a gradual expansion of the reasoning process (see Appendix D for examples).

The optimization characteristics also differ significantly in terms of gradient scale. As shown in Figure 5(b),FIPO’s Gradient Norm remains low and consistent throughout training, characterizing an evolutionbuilt upon fine-grained updates. In contrast, the baseline (DAPO) displays highly volatile fluctuations,with frequent, violent spikes in its gradient norm. These fluctuated updates indicate that DAPO’s searchprocess is at risk of abrupt shifts and potential instability.

This contrast in stability is further reflected in the policy entropy (Figure 5c). While FIPO maintains asmooth and sustained rise in entropy, indicating a continuous and stable exploration of the reasoning

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/e512e4b81a9283ba5a6f9326cbb2b21cd4fc4c491ad5c09a0ae85bd7a12121de.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/0ba01801b6429047be6ecadc346ddf48cdbea81be457c184a33f51ba9e027c3f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/952344ef5d8cb417641a8aa5312ed71470ee1eee43854c0b01462c7e9f0615ee.jpg)



Figure 4: Analysis of training reward and length-weighted advantages.(a) Mean training rewards.DAPO achieves higher raw scores, which is expected as both methods incorporate an overlong penaltythat suppresses the reward of longer responses. (b)Number of Sampled Batches. This metric indicatesthe sampling redundancy required to maintain a sufficient number of effective batches. A higher samplingneed suggests the model frequently generates non-informative trajectories on the training set, serving asa potential indicator of overfitting. (c) Response length weighted mean advantages. FIPO exhibits asustained upward trend, establishing a positive reinforcement cycle where longer responses increasinglyyield positive advantages. In contrast, DAPO shows a declining trend, suggesting a failure to convertlength into effective reasoning gains, which ultimately limits its performance.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/814db2cf4a2d3bcd32895530365fde75b127954b878d8e1f2df2897a13592294.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/6830b74bbc4c3ba46bfdfa48d48fe9b6a50b42fb909a7df98ada35d64d74011b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/2d57a42fb41b7b5d4db0cd6851ad004ca7a94d3111f59d8d692958bb9b6d8a21.jpg)



Figure 5: Policy evolution and optimization dynamics. (a) Policy KL divergence. (b) Policy Entropy. (c)Gradient Norm. FIPO exhibits a more controlled policy drift and smoother update gradients than DAPO.Notably, FIPO’s rising entropy, paired with the weighted advantage trends in Fig. 4(b), indicates that themodel is actively exploring a broader reasoning space where longer CoT paths increasingly correspondto correct solutions.


space, DAPO’s entropy is marked by noisy oscillations throughout the training process. In contrast,DAPO’s entropy is marked by noisy oscillations as training progressed. Together, these metrics depictFIPO as a model that achieves significant and purposeful policy evolution toward complex reasoningwhile ensuring the optimization process remains numerically well-behaved.

# 7 Conclusion

In this paper, we introduced Future-KL Influenced Policy Optimization (FIPO), a reinforcement learningapproach designed to resolve the coarse-grained credit assignment problem inherent in standard GRPO.By incorporating discounted Future-KL divergence into policy updates, FIPO transforms sparse outcome-based rewards into dense, token-level supervision. Our empirical analysis identifies and addresses acritical “length-performance plateau” in existing baselines, demonstrating that standard uniform rewardsfail to sustain long-chain reasoning. Validated on Qwen2.5-32B-Base, FIPO effectively breaks this ceiling:

it propels performance on AIME 2024 from a baseline of $5 0 . 0 \%$ to a peak of ${ \bf 5 8 . 0 \% }$ (converging at ${ \bar { 5 } } 6 . 0 \%$ )and extends the average chain-of-thought length from 4,000 to over 10,000 tokens. Crucially, thesefindings challenge the prevailing assumption that complex critic models are necessary for granularcredit assignment, proving that dense supervision can be effectively realized within the more efficientGRPO framework. To facilitate future research, we open-source our complete training code and recipes,providing the community with a scalable and accessible pathway to advance large-scale reasoningmodels.

# 8 Limitations and Future Work

Despite its effectiveness, FIPO has certain limitations:

Cost and Efficiency. A primary constraint is the increased computational cost associated with extendingreasoning sequences. As FIPO successfully unlocks CoT lengths exceeding 10,000 tokens, the trainingand inference overhead grows significantly, posing challenges for resource-constrained deployments. Weargue that the development of advanced reasoning should be a sequential process: first eliciting long,high-quality reasoning capabilities, and subsequently optimizing them for efficiency. While this paperfocuses on the first stage, breaking through length stagnation to achieve superior performance, the taskof transforming these long reasoning paths into more concise and efficient forms is a critical next step.We will leave this for future exploration.

Task Generalization. Another limitation is that our evaluations are primarily conducted on mathematicalreasoning benchmarks. However, we contend that mathematics serves as a rigorous and representativeproxy for deep reasoning; its requirement for objective, verifiable ground truth and high-density logicalconsistency makes it the most demanding testbed for our algorithm. Having demonstrated that FIPO canovercome length stagnation in this challenging domain, we leave the exploration and validation of theseelicited behaviors in other open-ended or less structured domains for future work.

Training Data Scope. To ensure a rigorous and fair comparison with the baseline, we restricted ourtraining exclusively to the dataset used in DAPO. Consequently, we have not yet explored the scalabilityof FIPO on larger-scale or higher-quality datasets. While this controlled setting serves to isolate thealgorithmic contributions of our method, the potential of FIPO when trained on more extensive ordiverse data distributions remains uncharted. Moreover, while FIPO achieves superior performanceover o1-mini on mathematical benchmarks, this advantage is inherently domain-specific. Given that ourtraining was strictly confined to the math dataset, we do not anticipate these gains to generalize acrossnon-mathematical domains, such as coding or symbolic logic, where o1-mini benefits from massive-scale,multi-stage reinforcement learning. Consequently, we leave the exploration of FIPO’s generalizationacross broader data regimes and its fundamental scaling properties for future work.

Limited Model Scope. A core objective of our study is to investigate RL-driven reasoning startingfrom a clean base model with no prior exposure to Long-CoT synthetic data. This strict requirement forexperimental purity significantly limits the selection of suitable backbone models. Most contemporaryopen-source models optimized for reasoning have already undergone extensive supervised fine-tuning(SFT) or distillation from long-form reasoning traces. We contend that the underlying training dynamicsof eliciting reasoning directly from a vanilla base model differ fundamentally from further optimizinga model that has already internalized distilled reasoning patterns. Consequently, our choice of modelswas restricted to a few high-quality vanilla base models, such as the Qwen2.5 series, to ensure that ourfindings specifically characterize the emergence of inherent reasoning potential rather than the refinementof pre-distilled CoT behaviors. In future work, we plan to investigate the efficacy and mechanisticbehavior of our algorithm when applied to such pre-distilled Long-CoT models, exploring whetherthe dense advantage formulation can further refine or synergize with pre-existing distilled reasoningcapabilities.

Performance Gap vs. Distillation. While RL-based self-evolution significantly enhances reasoning, itremains a "discovery-based" process that is inherently less efficient than direct distillation. Larger teachermodels provide a much denser supervisory signal and superior heuristics (logits) that are difficult for asmaller model to self-derive through sparse rewards alone, resulting in a persistent performance gapbetween self-trained and distilled variants.

# 9 Contributions

# Core Contributors

Chiyu Ma1,5, Shuo Yang2,5

# Contributors

Kexin Huang5, Jinda Lu5, Haoming Meng3,5, Shangshang Wang4,5

# Supervision

Bolin Ding6, Soroush Vosoughi1, Guoyin Wang5, Jingren Zhou6

# Affiliations

1 Dartmouth College

2 Peking University

3 University of Toronto

4 University of Southern California

5 Qwen Pilot Team

6 Alibaba

# References



Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. The unreasonable effectivenessof entropy minimization in llm reasoning. arXiv preprint arXiv:2505.15134, 2025.





Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon,Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier withadvanced reasoning, multimodality, long context, and next generation agentic capabilities. arXivpreprint arXiv:2507.06261, 2025.





Tiantian Fan, Lingjun Liu, Yu Yue, Jiaze Chen, Chengyi Wang, Qiying Yu, Chi Zhang, Zhiqi Lin, RuofeiZhu, Yufeng Yuan, et al. Truncated proximal policy optimization. arXiv preprint arXiv:2506.15050, 2025.





Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai,Jingren Zhou, and Junyang Lin. Soft adaptive policy optimization. arXiv preprint arXiv:2511.20347,2025.





Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang,Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning.Nature, 645(8081):633–638, 2025.





Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang,Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312,2025.





Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXivpreprint arXiv:2503.24290, 2025.





Kexin Huang, Haoming Meng, Jukang Wu, Jinda Lu, Chiyu Ma, Ziqian Chen, Xue Wang, Bolin Ding,Jiancan Wu, Xiang Wang, Xiangnan He, Guoyin Wang, and Jingren Zhou. On the direction of RLVRupdates for LLM reasoning: Identification and exploitation. In The Fourteenth International Conferenceon Learning Representations, 2025. URL https://openreview.net/forum?id=r6Pw3RiMYL.





Zhen Huang, Haoyang Zou, Xuefeng Li, Yixiu Liu, Yuxiang Zheng, Ethan Chern, Shijie Xia, Yiwei Qin,Weizhe Yuan, and Pengfei Liu. O1 replication journey–part 2: Surpassing o1-preview through simpledistillation, big progress or bitter lesson? arXiv preprint arXiv:2411.16489, 2024.





Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Hel-yar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprintarXiv:2412.16720, 2024.





Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman,Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in openlanguage model post-training. arXiv preprint arXiv:2411.15124, 2024.





Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin.Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.





Haoming Meng, Kexin Huang, Shaohang Wei, Chiyu Ma, Shuo Yang, Xue Wang, Guoyin Wang, BolinDing, and Jingren Zhou. Sparse but critical: A token-level analysis of distributional shifts in RLVRfine-tuning of LLMs. In The Fourteenth International Conference on Learning Representations, 2025. URLhttps://openreview.net/forum?id=8vWIXno8LW.





Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang,Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructionswith human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.





Yiwei Qin, Xuefeng Li, Haoyang Zou, Yixiu Liu, Shijie Xia, Zhen Huang, Yixin Ye, Weizhe Yuan, HectorLiu, Yuanzhi Li, et al. O1 replication journey: A strategic progress report–part 1. arXiv preprintarXiv:2410.18982, 2024.





John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensionalcontinuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.





John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policyoptimization algorithms. arXiv preprint arXiv:1707.06347, 2017.





Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, NathanLambert, Sewon Min, Ranjay Krishna, et al. Spurious rewards: Rethinking training signals in rlvr.arXiv preprint arXiv:2506.10947, 2025.





Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, MingchuanZhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in openlanguage models. arXiv preprint arXiv:2402.03300, 2024.





Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, HaibinLin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the TwentiethEuropean Conference on Computer Systems, pp. 1279–1297, 2025.





Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, DarioAmodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neuralinformation processing systems, 33:3008–3021, 2020.





Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao,Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXivpreprint arXiv:2501.12599, 2025a.





Ling Team, Anqi Shen, Baihui Li, Bin Hu, Bin Jing, Cai Chen, Chao Huang, Chao Zhang, Chaokun Yang,Cheng Lin, et al. Every step evolves: Scaling reinforcement learning for trillion-scale thinking model.arXiv preprint arXiv:2510.18855, 2025b.





Junkang Wu, Kexin Huang, Jiancan Wu, An Zhang, Xiang Wang, and Xiangnan He. Quantile advantageestimation for entropy-safe reasoning. arXiv preprint arXiv:2509.22611, 2025a.





Mingqi Wu, Zhihao Zhang, Qiaole Dong, Zhiheng Xi, Jun Zhao, Senjie Jin, Xiaoran Fan, Yuhao Zhou,Huijie Lv, Ming Zhang, et al. Reasoning or memorization? unreliable results of reinforcement learningdue to data contamination. arXiv preprint arXiv:2507.10532, 2025b.





Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang,Zhihao Zhang, Honglin Guo, et al. Bapo: Stabilizing off-policy reinforcement learning for llms viabalanced policy optimization with adaptive clipping. arXiv preprint arXiv:2510.18927, 2025.





An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu,Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert modelvia self-improvement. arXiv preprint arXiv:2409.12122, 2024.





An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.





Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, XipengWu, Qingwei Guo, et al. Mastering complex control in moba games with deep reinforcement learning.In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 6672–6679, 2020.





Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan,Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale.arXiv preprint arXiv:2503.14476, 2025.





Yufeng Yuan, Yu Yue, Ruofei Zhu, Tiantian Fan, and Lin Yan. What’s behind ppo’s collapse in long-cot?value optimization holds the secret. arXiv preprint arXiv:2503.01491, 2025.





Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang,TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advancedreasoning tasks. arXiv preprint arXiv:2504.05118, 2025.





Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin,Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXivpreprint arXiv:2508.06471, 2025.





Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reasonwithout external rewards. arXiv preprint arXiv:2505.19590, 2025.





Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu,Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.





Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang,Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084,2025.



# Appendix Table of Contents

# A Parameter Settings 20

A.1 Qwen2.5 32B Base 20

A.2 Qwen2.5 7B Math 20

# B Result on Qwen2.5 7B Math 21

B.1 Performance 21

B.2 Result Analysis 21

# C Ablation Studies 22

C.1 Clip-High, Max Length, and Response Length 23

C.2 Extreme Value Filtering . . 24

C.3 Influence Weight Clipping . . 25

C.4 Effective Horizon of Decay Rate . 26

# D Case Study 27

# E Discussions on failure trials and reproduction 28

# F More details on Training Cost 30

# A Parameter Settings

We detail the specific hyperparameter configurations employed for fine-tuning Qwen2.5-32B-Base andQwen2.5-7B-Math. We started from the publicly released DAPO training srcipt. To ensure full repro-ducibility, we will release our complete codebase along with the training scripts used in our experiments.

# A.1 Qwen2.5 32B Base

We conduct training using a global batch size of 512 with a group size of $G = 1 6$ generations per prompt.The model is optimized using a learning rate of $1 \times 1 0 ^ { - 6 }$ and a weight decay of 0.1. For the FIPO specificparameters, we set the Future-KL decay rate to 32.0 and employ a safety threshold of 10.0 to filter extremeinfluence weights. The policy updates are constrained with asymmetric clipping ratios of [0.2, 0.28]. Tosupport extensive reasoning chains, the maximum response length is set to 20,480 tokens. A detailedhyperparameter comparison between our method and DAPO can be found in Table 2.


Table 2: Hyperparameter settings for Qwen2.5-32B-Base experiments. We compare the configuration ofour proposed FIPO against the DAPO baseline. Most infrastructure and optimization settings remainidentical to ensure a fair comparison.


<table><tr><td>Hyperparameter</td><td>DAPO (Baseline)</td><td>FIPO (Ours)</td></tr><tr><td colspan="3">Shared Optimization Settings</td></tr><tr><td>Base Model</td><td colspan="2">Qwen2.5-32B-Base</td></tr><tr><td>Global Batch Size</td><td colspan="2">512</td></tr><tr><td>Group Size (G)</td><td colspan="2">16</td></tr><tr><td>Learning Rate</td><td colspan="2">1e-6</td></tr><tr><td>LR Scheduler</td><td colspan="2">Constant with 10 Warmup Steps</td></tr><tr><td>Weight Decay</td><td colspan="2">0.1</td></tr><tr><td>Gradient Clipping</td><td colspan="2">1.0</td></tr><tr><td>Max Prompt Length</td><td colspan="2">2,048</td></tr><tr><td>Max Response Length</td><td colspan="2">20,480</td></tr><tr><td>Overlong Buffer</td><td colspan="2">4096</td></tr><tr><td>Sampling Temp / Top-p</td><td colspan="2">1.0 / 1.0</td></tr><tr><td>Dual Clip Ratio</td><td colspan="2">10.0</td></tr><tr><td>Policy Clip Ratio</td><td colspan="2">[0.2, 0.28] (Asymmetric)</td></tr><tr><td>KL Penalty Coef</td><td colspan="2">0.0</td></tr><tr><td colspan="3">Method-Specific Settings</td></tr><tr><td>Mini-Batch Size</td><td>32</td><td>64</td></tr><tr><td>Loss Function</td><td>DAPO</td><td>Future-KL</td></tr><tr><td>Future-KL Decay Rate</td><td>-</td><td>32.0</td></tr><tr><td>Future-KL Clip Ratio</td><td>-</td><td>[1.0, 1.2]</td></tr><tr><td>Safety Threshold</td><td>-</td><td>10.0</td></tr></table>

# A.2 Qwen2.5 7B Math

We initially validated the effectiveness of our approach on Qwen2.5-7B-Math as a pilot study before scal-ing to the 32B parameter regime. During this preliminary phase, we observed that training performancewas initially volatile, and the reproducibility of reasoning gains was inconsistent across independentruns. To address these stability issues, we increased the group size to $G = 3 2$ to provide a more stableoptimization signal and enforced a stricter advantage clipping threshold of 3.0 to filter out destabilizingupdates. These adjustments successfully stabilized the training trajectory, making the results morereliable while maintaining the same performance level. The model was optimized using a learningrate of $1 \times 1 0 ^ { - 6 }$ and a weight decay of 0.1, with the Future-KL decay rate set to 32.0. Following thestabilization of these pilot experiments, we scaled the validated framework to the 32B model. A summaryof the 7B hyperparameter settings is provided in Table 3, and detailed results of the 7B MATH experimentsare available in Sec. B.


Table 3: Hyperparameter settings for Qwen2.5-7B-MATH experiments. We compare the configurationof our proposed FIPO against the DAPO baseline. Most infrastructure and optimization settings remainidentical to ensure a fair comparison.


<table><tr><td>Hyperparameter</td><td>DAPO (Baseline)</td><td>FIPO (Ours)</td></tr><tr><td colspan="3">Shared Optimization Settings</td></tr><tr><td>Base Model</td><td colspan="2">Qwen2.5-7B-MATH</td></tr><tr><td>Global Batch Size</td><td colspan="2">512</td></tr><tr><td>Learning Rate</td><td colspan="2">1e-6</td></tr><tr><td>LR Scheduler</td><td colspan="2">Constant with 10 Warmup Steps</td></tr><tr><td>Weight Decay</td><td colspan="2">0.1</td></tr><tr><td>Gradient Clipping</td><td colspan="2">1.0</td></tr><tr><td>Max Prompt Length</td><td colspan="2">2,048</td></tr><tr><td>Sampling Temp / Top-p</td><td colspan="2">1.0 / 1.0</td></tr><tr><td>Dual Clip Ratio</td><td colspan="2">10.0</td></tr><tr><td>Policy Clip Ratio</td><td colspan="2">[0.2, 0.28] (Asymmetric)</td></tr><tr><td>KL Penalty Coef</td><td colspan="2">0.0</td></tr><tr><td colspan="3">Method-Specific Settings</td></tr><tr><td>Mini-Batch Size</td><td>32</td><td>64</td></tr><tr><td>Loss Function</td><td>DAPO</td><td>Future-KL</td></tr><tr><td>Future-KL Decay Rate</td><td>-</td><td>32.0</td></tr><tr><td>Future-KL Clip Ratio</td><td>-</td><td>[0.8, 1.2]</td></tr><tr><td>Safety Threshold</td><td>-</td><td>3.0</td></tr><tr><td>Group Size (G)</td><td>16</td><td>32</td></tr><tr><td>Max Response Length</td><td>8192</td><td>10240</td></tr><tr><td>Overlong Buffer</td><td>4096</td><td>2048</td></tr></table>

# B Result on Qwen2.5 7B Math

This section documents the experimental results on the Qwen-2.5 7B MATH model. Due to the pro-hibitive computational cost of 32B models, we conducted the majority of our early-stage explorationsand ablations using this variant. As noted in Section 3.2, the 7B model exhibited noticeable performancesensitivity across different training configurations. To address this, we performed targeted hyperparame-ter adjustments to stabilize the training dynamics and ensure the reliability of the observed trends. Notethat we extend the context window from 4096 to 32768, following the instructions from DAPO verl scripts.

# B.1 Performance

Table 4 presents a comparative analysis of Pass@1 performance across different RL-based methods onthe AIME 2024 and AIME 2025 benchmarks. Our proposed FIPO (7B) achieves a notable performanceof $4 0 . 0 \%$ on AIME 2024, significantly outperforming the GRPO (7B) baseline $( 2 2 . 0 \% )$ and the DAPO(7B) method $( 3 6 . 0 \% )$ . However, we observe a general performance compression on the AIME 2025benchmark. While FIPO still maintains a leading edge at $1 9 . 0 \%$ , the performance gap between the threemethods narrows considerably compared to the AIME 2024 results. This phenomenon is primarilyattributed to the substantially higher intrinsic difficulty and the "live" nature of the AIME 2025 problems,which appear to approach the reasoning ceiling of 7B-parameter models without external guidance. Atthis scale, especially in the absence of pre-training on more advanced chain-of-thought structures, theincreased complexity of the reasoning chains required for AIME 2025 makes it challenging to distinguishthe marginal gains of different optimization algorithms, as most models encounter similar structuralbottlenecks.

# B.2 Result Analysis

As shown in Figure 6, both algorithms maintain a stable length-weighted mean advantage and a meanresponse length fluctuating around 1200 tokens. Notably, this stands in sharp contrast to the 32B modeltraining, where FIPO typically triggers a sustained growth in response length. We hypothesize that this


Table 4: Qwen 2.5-7B-MATH Performance Comparison of Pass@1 Performance on AIME2024 andAIME2025. All results are reported as percentages $( \% )$ . We report the peak average Pass@1 across 32samples $( \mathrm { A v g @ } \ @ 3 2 )$ . To reduce sensitivity to digit-level generation variance, final values are rounded tothe nearest integer.


<table><tr><td>Method</td><td>AIME 2024 (Pass@1)</td><td>AIME 2025 (Pass@1)</td></tr><tr><td>GRPO (Guo et al., 2025)</td><td>22.0%</td><td>18.0%</td></tr><tr><td>DAPO</td><td>36.0%</td><td>18.0%</td></tr><tr><td>FIPO (Ours)</td><td>40.0%</td><td>19.0%</td></tr></table>

length stagnation is not an optimization failure, but rather a reflection of the 7B model’s inherent capacitylimits and training priors. Specifically, the Qwen-2.5 7B MATH base was pre-trained with a restricted 4Kcontext window (Yang et al., 2024), which likely imposes a physical ceiling on its reasoning depth withoutexternal guidance. Furthermore, the model’s strong initial bias toward code-based reasoning (Shao et al.,2025), which favors logically dense, deterministic pathways over verbose exploration, combined withpotential AIME24 data leakage (Wu et al., 2025b), provides "high-confidence shortcuts" that allow themodel to reach concise solutions quickly rather than exploring longer, iterative paths.

This phenomenon of "efficient but constrained" reasoning is further corroborated by the entropy dynamicsshown in Figure 7(c). While larger models typically rely on sustained entropy growth to explore complexreasoning spaces, the 7B MATH model achieves optimal performance by converging onto a markedlylower entropy policy under FIPO. This divergence suggests a fundamental difference in scaling behavior:while the 32B model benefits from broad exploration, the 7B model appears to prioritize the refinementof a specific, high-confidence reasoning manifold.

The hypothesis that a low-entropy state is critical for performance at this scale is further supported byour ablation study on the Future-KL influence weight clipping. As discussed in Appendix C, the 7B and32B models exhibit different sensitivities to the Future-KL clipping range. While the 32B model maintainssuperior performance with an influence weight clipped between 1.0 and 1.2, applying this same range tothe 7B model leads to a continuous increase in entropy, similar to the behavior of the 32B model, yet resultsin a significant degradation in performance. Instead, we found that the 7B model performs optimallyunder a different regime where a clipping range of 0.8 to 1.2 is applied. This observation suggests thatthe 7B model may lack sufficient inherent self-exploration capacity to derive benefit from higher-entropystates. In this regime, maintaining higher entropy appears to introduce more detrimental noise thanuseful discovery. Consequently, at this scale, superior performance is most likely attainable whenthe model converges to specific, low-entropy reasoning traces. This phenomenon is fundamentallyconsistent with the principles of entropy minimization (Agarwal et al., 2025), entropy regulation (Wuet al., 2025a), self-guiding (Zuo et al., 2025), and self-certainty optimization (Zhao et al., 2025), whichhave typically been observed to be effective in models of similar scale.

# C Ablation Studies

In this section, we provide a comprehensive analysis of the ablation studies conducted to validate ourmethodology. We first examine the impact of high-value clipping and the extension of the maximumresponse length on the resulting mean response length during 32B model training with FIPO. Next, wepresent results on the 7B model scale regarding weight filtration, which complement the stability-relatedfindings observed in our 32B experiments. Furthermore, we investigate the implications of variousadaptive clipping configurations and their influence on performance, specifically highlighting how thesechoices drive divergent behaviors in policy entropy. Lastly, we evaluate the sensitivity of the optimizationprocess to different decay rate options and their potential impact on convergence dynamics.


(a) Length Weighted Mean Advantage


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/ad825a7ea34e497bf94250362a775faf7d659d07556c28559266d32e0419b739.jpg)



(b) Mean Response Length in Tokens


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/2e666d7100ba891586117c8f4afde55a51741722824af3be9222c2b19dbcd005.jpg)



Figure 6: Analysis of length-weighted advantages and mean response length of Qwen2.5-7B-MATH.(a) Subfigure (a) shows FIPO maintains a stable Length-Weighted Mean Advantage. (b) Subfigure (b)shows the mean response length remains steady around 1200 tokens. This suppressed length growth islikely caused by the 4K context window limit of the base model, and its inherent bias toward concisecode-based reasoning, which anchors the generation depth.



(a) Policy KL


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/57de930235e76018570ac839d5a9ea3477c9e1156ede19b9da8e8c8b67359b2a.jpg)



(b) Gradient Norm


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/b5daee6d9f657683cf98f8be000fd663af41f4c28af0f3f59e6ad5544cf63de4.jpg)



(c) Entropy


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/aa26b1332af7641120f1f397f78cc4d33ab2f9f5963ab4dbcd22b8c5e78a249f.jpg)



Figure 7: Policy evolution and optimization dynamics of Qwen2.5-7B-MATH. (a) Policy KL and (b)Gradient Norm illustrate the optimization path. (c) Notably, FIPO results in lower entropy compared toDAPO, indicating that the model converges toward more deterministic reasoning traces during the RLprocess.


# C.1 Clip-High, Max Length, and Response Length

One of the most interesting phenomena we observed is an unexpected surge in response length at theearly stage when turning up the clip-high ratio $\epsilon _ { h i g h }$ of the PPO objective. In this study, we set $\epsilon _ { h i g h } = 1 . 4$providing a larger trust region for policy updates when the advantage is positive. For the maximumlength trial, we extended the limit to 25K tokens (from the default 20K) while maintaining the overlongbuffer at a consistent $2 0 \%$ ratio. All other variables remained consistent to ensure a fair comparison.As shown in Fig. Figure 8(a-d), a higher $\epsilon _ { h i g h }$ allows the policy to deviate more aggressively from theold distribution when receiving positive advantages, incentivizing the model to "over-explore" longerreasoning paths almost immediately. However, this surge comes at the cost of training stability; asdepicted in Figure 8(e), the policy entropy explodes under a higher $\epsilon _ { h i g h } ,$ indicating that the optimizationlandscape becomes volatile when the clipping constraint is too relaxed.

A strikingly similar failure mode is observed when increasing the maximum response length limit.While this also triggers a rapid growth in token count, it reveals a clear diminishing return in reasoningefficiency. Both cases suffer from the emergence of superficial reasoning behaviors, such as repetitivecontent, task-irrelevant LaTeX formatting, and premature self-reflection. We contend that introducing

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/03d679308aaa6a0a1ca67d409362321125e74a55ff9b4471e87c86c6e03fecd1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/e9682cf2ea89d1031149883c5c64e452dae1393b1ed9959073643c38ce56c136.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/dd642750b63a2253ce129ecbba47d127352f03d7db4afdeb97f80f61eaf30e1d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/3892fabc0f5310f5185a9f33007da82c0998e34bf615ca52a30c689431fb23fc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/e90ce92a86e361e9762e61860e6f5f92c8db78e509e61617d081678bbde46433.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/7a83b19cf649daaa75b8281082fd75a2447bbd116561c672fa0e9f13a326fc37.jpg)



Figure 8: Ablation study on Response Length Control on Qwen2.5-32B-Base. We compare the baselineFIPO with FIPO under cliphigh 1.4 and FIPO with a max resonse length of 25K. (a-d) Increasing themaximum length limit triggers a significant early surge in response length, yet results in lower trainingefficiency and marginal performance gains. (e) A higher clip ratio causes a sharp escalation in policyentropy, indicating potential training instability. (f) Overall, while both modifications encourage longerresponses, they are less effective in improving AIME24 performance compared to the balanced FIPOconfiguration.


self-reflection can be counterproductive when the model has not yet established a solid foundation forone-step reasoning. In such cases, the model lacks the verifiable internal logic needed to evaluate itsown state, leading to internal consistency conflicts, a phenomenon where the model oscillates betweenconflicting steps without progressing toward the correct solution. These observations suggest that theemergence of higher-order reasoning patterns, such as self-reflection, is not necessarily "the earlier,the better" during the training process. Instead, it should be a progressive maturation where complexstrategies build upon a stable base of fundamental logic. Consequently, the performance gain on AIME24(Figure 8f) is marginal compared to the balanced FIPO baseline. These results underscore the importanceof governed growth in response length, ensuring that extended computation is allocated to genuinelogical depth rather than redundant self-correction cycles.

# C.2 Extreme Value Filtering

As we noted in subsection 4.2, since the computation of Future-KL is highly dependent on the logitvalues between each update, one of its drawbacks is its sensitivity to importance ratio volatility. If thevariation in importance ratios becomes significant, the Future-KL weight can become highly volatile atthe same time. Rather than providing a stable guiding signal for policy updates, the excessive noise inimportance ratios causes the Future-KL term to lose its effectiveness. In such cases, the incorporation offuture KL information could result in less stable updates, and the model’s performance tends to revert toa suboptimal level close to the standard baseline without showing the intended improvements. This isparticularly observable in 7B models, which exhibit frequent fluctuations in importance ratios.

Table 5 presents the ablation results on the 7B model. We observe that without filtering extreme ISratios, the performance gain on AIME2024 is suboptimal compared to the configuration where filtering isapplied. Although the unfiltered version achieves a slightly higher score on AIME2025 in one instance,the overall reliablity and the peak performance on AIME2024 $( 4 0 . 0 \% )$ along with the observations made

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/f60c64dd4270e56ec5c1405b0aa34b77df34722f3526744ff210963873eb9362.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/838c7afc87ecb5907184f4e5d05ed9804510c3884b5fbda86b0069bc2a7af191.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/7db21d868642f41a6db6d31bcd107455ab95a5c3983e204fe04f8384350b6932.jpg)



Figure 9: Ablation Study on Filtering Mechanism in Future-KL on Qwen2.5-7B-MATH. We comparethe training stability of FIPO with and without filtering extreme importance sampling (IS) ratios. (a)Future-KL influence weights: removing extreme IS ratios results in a significantly more constrainedand compact weight range, leading to more stable policy updates. (b-c) Clipping statistics: extreme ISratios rise disproportionate influence weights that drive more aggressive updates. This is reflected in theincreased dual clip and policy clip fractions, where a larger proportion of tokens are pushed beyond thetrust region when filtering is absent.


on 32B training confirm the necessity of the filtering mechanism for a more reliable training.

Moreover, as shown in Figure 9(a), removing extreme IS ratios results in a significantly more constrainedand compact weight range, which provides a more stable signal for the advantage estimator. In contrast,the absence of filtering leads to disproportionately large influence weights. This leads to a higherfrequency of aggressive policy updates, as evidenced by the clipping fractions in Figure 9(b) and (c). Whenfiltering is absent, a larger proportion of tokens are pushed beyond the designated trust region, resultingin higher dual clip and policy clip fractions. This suggests that the model is frequently attempting tomake updates that exceed the constraints of the PPO objective, thereby explaining the suboptimal gainsin performance when IS ratio volatility is high.


Table 5: Ablation on Extreme Value Filtering and Influence Weight Clipping Performance Comparisonof Pass@1 Performance on AIME2024 and AIME2025 on Qwen2.5-7B-MATH. All results are reported aspercentages $( \% )$ . We report the peak average Pass@1 across 32 samples $( \mathrm { A v g @ } \ @ 3 2 )$ .


<table><tr><td>Method</td><td>AIME 2024 (Pass@1)</td><td>AIME 2025 (Pass@1)</td></tr><tr><td>FIPO (εflow=1.0, εfhigh=1.2)</td><td>36.0%</td><td>19.0%</td></tr><tr><td>FIPO (w/o filtering, εflow=0.8, εfhigh=1.2)</td><td>38.0%</td><td>21.0%</td></tr><tr><td>FIPO (εflow=0.8, εfhigh=1.2)</td><td>40.0%</td><td>19.0%</td></tr></table>

# C.3 Influence Weight Clipping

Table 5 shows the result of ablations on the influence weight clipping. We primarily modified the clippingrange with respect to 1. An influence weight with a range of [1.0, 1.2] essentially provides more penaltyfor negative samples and more reward to positive ones. The range of [0.8, 1.2] inherits these propertieswhile offering a more balanced influence; it further reduces the reward when a token in a positivesample is associated with subsequent negative behaviors, and reduces the penalty when a token in anegative sample is associated with subsequent positive behaviors. This helps to provide a more controlledexploration and thus resulted in the performance improvements shown in Table 5.

Specifically, the [0.8, 1.2] configuration achieves $4 0 . 0 \%$ on AIME 2024, surpassing the $3 6 . 0 \%$ of the [1.0, 1.2]setting. While the latter encourages more aggressive exploration, as evidenced by the continuous growth

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/2634027fd37b3f903bf5453e7c1b46a9687148dca4efc3ba35087a79641670b4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/0d36d3f3a1e5a2ad722e2f2de3ef568e63809b22de9be1e0897a5fadcb7287b8.jpg)



Figure 10: Ablation Study on Influence Weight Clipping on Qwen-2.5-7B-MATH. (a) Similar to 32B,a influence weight with range [1.0, 1.2] resulted in a continuous growth in entropy, thus encouragingmore explorations. (b) However, we do not observe any surge in response length as we discussed in theprevious section.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/bf290defd7995a6876eb099d83676e62a9c8dd6b2e936bb62ff78135cf16a9b0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/059a71d516b331842eff8b2a74c7c2c84bad2fdd46253c73fa9627bbbb60d7e4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/decde2a3a36897c4ec09a230689fc5999e50d5aeb2bcfc81c18b8d92ec295921.jpg)



Figure 11: Ablation Study on the Decay Rate Horizon $\mathbf { \rho } ( \tau )$ on Qwen2.5-7B-MATH. (a) Mean InfluenceWeight: Extending the decay horizon (e.g., $\tau = 2 5 6$ ) significantly increases weight fluctuations, potentiallydestabilizing policy updates. Conversely, a highly restrictive horizon (e.g., $\tau = 8$ ) limits the weights tooclose to the baseline. (b) Influence Weight Clip Ratio: A larger decay horizon results in a higher clippingfrequency. This reflects more drastic variations in the raw influence weights, directly corroborating thetrends observed in (a). (c) Entropy: An intermediate horizon $\tau = 3 2$ ) strikes a critical balance. It avoidsthe excessive volatility induced by larger decay rates, as well as the premature entropy collapse causedby the myopic guidance of shorter horizons, allowing the 7B model to maintain controlled and effectiveexploration.


in policy entropy shown in Figure 10(a), our results suggest that 7B models are sensitive to such excessiveexploration pressure. This aligns with our broader observation on scaling behavior: unlike larger modelsthat benefit from broad exploration, the 7B model appears to prioritize the refinement of a specific,high-confidence reasoning manifold. In this context, the higher entropy induced by the [1.0, 1.2] rangeintroduces more detrimental noise than useful discovery. By contrast, the balanced influence of [0.8, 1.2]facilitates convergence toward a lower-entropy state, optimizing for self-certainty rather than stochasticsearch.

# C.4 Effective Horizon of Decay Rate

To complement our analysis, we further perform an ablation study on the effective horizon of the decayrate, specifically testing values of $\tau \in \{ 8 , 3 2 , 1 2 8 , 2 5 6 \}$ . These values define the half-life of the influenceweight decay, determining the token-distance over which future KL information remains significant.

Table 6 presents the performance across these configurations, while Figure 11 illustrates the correspondingtraining dynamics. As shown in Figure 11(a), the magnitude of fluctuation in Mean Influence Weightincreases as the decay horizon $\tau$ is extended. For the largest horizon $\tau = 2 5 6$ ), the weight mean exhibitsthe most significant deviation from 1.0, which can introduce instability into the policy updates. Incontrast, for a very small horizon $\left( \tau = 8 \right)$ , the influence weight remains highly proximal to 1.0, effectivelycausing the model to stay close to the baseline and relatively underutilize future KL information.

The entropy dynamics in Figure 11(c) further corroborate our earlier observations regarding the explo-ration capacity of the 7B model. While all configurations follow a characteristic trend of an initial decreasefollowed by a rise and eventual decline, the overall entropy levels are markedly higher for larger decayrates. The $\tau = 2 5 6$ setting maintains the highest entropy throughout most of the training, indicatingaggressive exploration driven by long-range future signals. Just as we observed with the restrictive[1.0, 1.2] influence weight range, this sustained high-entropy state introduces excessive volatility that the7B model struggles to accommodate. Conversely, the rapid entropy drop observed in the $\tau = 8$ settingrepresents an entirely different failure mode: because its influence weights fluctuate only marginally, themodel receives highly myopic guidance. Lacking sufficient long-term foresight, the policy prematurelycollapses into a suboptimal, low-entropy state. The intermediate horizon, $\tau = 3 2$ , strikes a criticalbalance. Similar to the stabilizing effect of the [0.8, 1.2] clipping range, $\tau = 3 2$ provides enough localizedfuture signal to safely navigate the updates without causing premature stagnation, allowing the policy toultimately converge onto a specific, high-quality reasoning manifold. This alignment across differenthyperparameters solidifies our hypothesis: at the smaller scale such as 7B, optimizing for self-certaintythrough controlled, moderate exploration is more effective than forcing broad, high-entropy searches.


Table 6: Ablation on the Effective Horizon of Decay Rate Performance Comparison of Pass@1 Perfor-mance on AIME2024 and AIME2025 on Qwen2.5-7B-MATH. All results are reported as percentages $( \% )$We report the peak average Pass@1 across 32 samples (Avg@32).


<table><tr><td>Method</td><td>AIME 2024 (Pass@1)</td><td>AIME 2025 (Pass@1)</td></tr><tr><td>FIPO (τ=8)</td><td>40.0%</td><td>17.0%</td></tr><tr><td>FIPO (τ=32)</td><td>40.0%</td><td>19.0%</td></tr><tr><td>FIPO (τ=128)</td><td>39.0%</td><td>21.0%</td></tr><tr><td>FIPO (τ=256)</td><td>42.0%</td><td>16.0%</td></tr></table>

# D Case Study

In this section, we provide a qualitative analysis of the model’s reasoning evolution by examining specificoutputs on the AIME 2024 competition. We randomly selected the responses from the grouped samples.The transition from DAPO’s length stagnation to FIPO’s sustained scaling is not merely a quantitativechange in token count, but a qualitative transformation in how the model utilizes its "thinking" budget.

Stage 1: Superficial Planning (Initial Step). As illustrated in Figure 13, the model at its initial stepexhibits "superficial planning" behavior. It produces a template-like outline of the solution steps but failsto execute the actual mathematical derivation. This results in a brief response that lacks logical substanceand often leads to an immediate hallucinated conclusion.

Stage 2: Linear Execution (DAPO Convergence & Early FIPO). Across its entire training duration,DAPO remains situated within this stage (Figure 14), evolving into a "linear executor" that can accuratelyfollow a standard Chain-of-Thought (CoT) to reach the ground truth. However, its reasoning is inherentlylimited to a "single-pass" logic, where the generation terminates immediately upon finding the first result.This explains the length stagnation observed in Figure 3 for DAPO.

Stage 3: Emergent Self-Reflection (Intermediate Stage FIPO). As FIPO training progresses to theintermediate stage (Figure 15), a distinct behavioral shift occurs. The model begins to utilize the expandingtoken budget (growing in reponse length) for spontaneous self-reflection. After deriving an initialresult, the model proactively initiates a verification phase, exploring alternative methodologies, such asswitching from algebraic manipulation to geometric interpretation, to cross-validate its findings.

Stage 4: Systematic Deep Reasoning (Late Stage FIPO). In the later stages of training (Figure 16), themodel matures into a "compute-heavy" strategy that prioritizes analytical depth. The reasoning trajectoryextends beyond simple reflection into a systematic audit. The model performs multiple passes of symbolicre-derivation and granular arithmetic verification (e.g., manually expanding large squares and squareroots step-by-step). This spontaneous emergence of self-verification aligns with the inference-time scalingbehaviors observed in advanced reasoning models, where the model treats length as a vital resource toensure better performance.

# E Discussions on failure trials and reproduction


(a) Response Length Mean in Tokens


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/e5118a13057b53da12f108fe3ead1bc6724771f8db85cb286ff06b199b8c119e.jpg)



(b) Policy Clip Fraction


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/7781c0ef19693e2eced5c8863167850fb31c55030a1a8c7d87a70a8d4116fe18.jpg)



(c) Gradient Norm


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/dcb915498e8a3ecc70f22325172d7dadaf5a7f4c1a390b16036405ccb6763e18.jpg)



(d) Entropy


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/be6fb8488bfdd8075cf4f7946688e36c48d718dc1c38dbc234bc0730ebb6c137.jpg)



(e) AIME24 Mean@32


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/d885ba0d83a07dae47cc7d5e4eb8ca41c9badc70be022b120d1d99c92e645e5d.jpg)



(f) AIME24 Cons@32


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-02/6f129ae0-e6ef-48f1-9d5d-b2d3fc753ed9/eb42f513be5ccddc3ffbde5cdca06c3de359380089c4746ab89cfb2ec599b22b.jpg)



Figure 12: Training Dynamics and Stability across Mini-batch Sizes on Qwen2.5-32B-Base. (a) Re-sponse Length Mean: While a mini-batch size of 32 can occasionally yield strong performance, it suffersfrom severe reproducibility issues and frequently encounters length stagnation. Notably, simply remov-ing the overlong penalty (w/o Overlong) fails to resolve this bottleneck. (b) Policy Clip Fraction: Inthe successful mini-batch 32 trial, fewer tokens are clipped compared to the failed cases, a phenomenonthat aligns with our findings in the clip_high ablation study. Because a mini-batch size of 64 is closerto on-policy and suffers less from importance sampling (IS) weight fluctuations, it naturally results infewer clipped tokens, thereby easing the scaling of response length and performance. (c) Gradient Norm:Similarly, the successful mini-batch 32 run maintains a noticeably stabler and slightly lower gradientnorm than the failed ones, a critical stabilizing property inherently achieved by the mini-batch 64 baseline.(d) Policy Entropy: The successful case sustains higher entropy throughout the training, indicatinghealthier exploration capabilities. (e-f) AIME24 Performance: Benefiting from stable optimization andsustained exploration, the successful mini-batch 32 case reaches a peak Mean@32 of approximately$6 0 \%$ and a Cons@32 of $7 0 \%$ , which is highly comparable to the final performance of our mini-batch 64baseline.


As noted earlier, we utilize a mini-batch size of 64 instead of 32 for our 32B model training and subsequent

experiments due to its relative stability. To better understand this parameter choice, we detail belowthe specific training results and issues we encountered when employing a mini-batch size of 32. Asillustrated in Figure 12(a), while a mini-batch size of 32 can occasionally yield strong performance, itsuffers from severe reproducibility issues. The most prominent failure mode is a severe decelerationin length growth, where the model struggles to effectively expand into longer reasoning trajectories.Notably, simply removing the overlong penalty fails to resolve this bottleneck, suggesting that the issuestems from deeper optimization dynamics rather than straightforward reward penalization.

We further interpret these findings through the underlying principles identified in our clip high ablationstudy (see subsection C.1). Although that ablation focuses on the clipping threshold rather than mini-batch size, it reveals a core optimization dynamic: when more tokens effectively participate in thepolicy update without being restricted by the clip boundary, the model gains significant momentumto expand its response length. Examining the policy clip fraction (Figure 12(b)), we observe that thesuccessful mini-batch 32 trial clips relatively fewer tokens than the failed ones. We deduce that whenimportance sampling (IS) weight fluctuations are reduced, a comparatively larger proportion of tokensremains unclipped and is fully included in the training signal. This valid inclusion is highly beneficial forovercoming the aforementioned length growth deceleration. Because a mini-batch size of 64 computesgradients over a broader sample set, it intrinsically stays closer to on-policy and suffers less from ISvariance. As a result, it naturally avoids excessive token clipping, providing the sustained optimizationmomentum required to smoothly scale the reasoning length and performance.

The differences in optimization stability are also evident in the gradient norm and policy entropy. Asshown in Figure 12(c), the successful mini-batch 32 run maintains a noticeably stabler and slightly lowergradient norm than the failed runs. Furthermore, it sustains higher entropy throughout the trainingprocess (Figure 12(d)), indicating healthier and more effective exploration capabilities without erraticdivergence. Crucially, these stabilizing properties, controlled gradient updates and sustained, activeexploration, are inherently achieved by the mini-batch 64 baseline, removing the need to rely on thestochastic success of smaller batch sizes. Ultimately, benefiting from stable optimization and consistentexploration, the successful mini-batch 32 case reaches a peak Mean@32 of approximately $6 0 \%$ and aCons@32 of $7 0 \%$ (Figure 12(e-f)). These metrics are highly comparable to the consistent performanceof our mini-batch 64 baseline. Therefore, our adoption of the 64 mini-batch size is not driven by ahigher theoretical performance ceiling, but rather by its capacity to reliably navigate the policy towards ahigh-quality reasoning manifold without falling into optimization traps.

It is crucial to distinguish this phenomenon from existing adaptive clipping techniques(Xi et al., 2025; Gaoet al., 2025). While adaptive clipping dynamically adjusts the threshold primarily to maintain trust-regionstability and prevent policy collapse, our findings highlight a fundamentally different mechanism: theeffective clip fraction acts as a structural valve for length expansion. By utilizing a mini-batch size of 64to naturally stabilize IS weights, we allow a larger proportion of valid tokens to safely pass through afixed boundary. Moreover, it is exactly under these stabilized optimization conditions that our proposedFuture KL mechanism can fully exert its intended effect as the primary driver for reasoning training.By design, Future KL incorporates forward-looking signals to guide current token updates, therebynaturally encouraging the exploration of extended reasoning trajectories. However, this mechanisminherently relies on a continuous and intact flow of gradients across long sequences. The stabilized ISweights provides this exact foundation: by preventing excessive token truncation, it ensures that thelong-term exploratory signals from Future KL are successfully and smoothly propagated backwardwithout being artificially interrupted. Supported by this unhindered optimization momentum, FutureKL explicitly pushes the policy to expand into deeper, high-quality reasoning manifolds, effectivelypreventing the model from collapsing back into premature, short-form responses.

# F More details on Training Cost

A naive implementation of Future KL requires computing a dense $\left( L , L \right)$ temporal decay matrix (where$L$ is the response length), resulting in an $\mathcal { O } ( L ^ { 2 } )$ memory footprint that easily causes Out-Of-Memory(OOM) errors during long-trajectory reasoning training. To mitigate this, we implemented a chunk-basedmemory-efficient algorithm, as shown in Listing 1. By partitioning the response sequence into blocks of afixed chunk size $( K )$ , we incrementally compute the distance masking and decay weights. The block-wisecontributions are computed via parallel matrix multiplications of shape $( B , K ) \times ( K , L )$ . This algorithmpreserves the exact analytical outcome of the Future KL formulation while strictly bounding the peakmemory complexity to $\mathcal { O } ( B \cdot L + L \cdot K )$ . Although the time complexity remains $\mathcal { O } ( B \cdot L ^ { 2 } )$ , the tensorizedblock operations are highly optimized on modern GPUs, effectively removing the memory bottleneck forscaling up reasoning lengths.

While standard GRPO computes the policy objective using strictly element-wise operations with an$\mathcal { O } ( B \cdot L )$ time complexity, the integration of our Future KL mechanism naturally introduces an $\mathcal { O } ( B \cdot L ^ { 2 } )$temporal aggregation process. Consequently, this imposes a certain degree of computational overheadduring the actor update phase. However, our chunked matrix multiplication implementation effectivelyvectorizes these operations, heavily leveraging the highly optimized dense MatMul capabilities of modernGPUs. Empirically, the wall-clock slowdown during the GRPO training iteration is relatively marginal andentirely acceptable. Most importantly, we argue that this modest trade-off in training speed is justifiable:it provides the dense, long-horizon credit assignment necessary to scale up complex reasoning, afundamental bottleneck that standard $\mathcal { O } ( B \cdot L )$ GRPO inherently struggles to overcome. Naturally,while our chunked implementation effectively addresses the memory constraints, more sophisticatedcomputational optimizations remain possible.


Listing 1: Python code for computing memory-efficient chunked Future KL


import torch   
def compute_future_kl_chunked(D,M,K,tau): #Calculate the decay factor gamma  $= 2\ast \ast$  (-1 / tau) #Mask out invalid tokens and filtered anomalies D  $= \mathrm{D}*\mathrm{M}$  # Initialize the Future KL accumulator F  $=$  torch.zeros_like(D) L  $= \mathrm{D}$  .shape[1] #Column vector of query positions (L x 1) i  $=$  torch.arange(L, device  $\equiv$  Ddevice).unsqueeze(1) for j_start in range(O,L,K): j_end  $= \min (\mathrm{j\_start + K},\mathrm{L})$  #Row vector of chunk positions (1 x Kcur) j  $=$  torch.arange(j_start,j_end,device  $\equiv$  Ddevice).unsqueeze(0) # Broadasted distance matrix (L x K.cur) Delta  $= \mathrm{j - i}$  # Decay weight block W  $=$  (gamma \*\*torch.clamp(Delta,min=O)) \* (Delta >= 0).float() # Extract KL values for the current chunk V  $= \mathrm{D}[:$  ,j_start:j_end] # Parallel matrix multiplication update F  $+ =$  torch/matmul(V,W.T) return F

# Question:

Find the largest possible real part of $[ ( 7 5 + 1 1 7 \mathrm { i } ) z + ^ { \cdot }$ \frac{96+144i}{z}]where $\$ 7$ is a complexnumber with $\$ 123,45$

Ground Truth: 540

# Response:

Step 1: The problem requires finding the largest possble real part of a complex expressioninvolving a complex number $\$ 7$ with a given magnitude.

Step 2: First, we need to express $\$ 7$ in polar form as $z = 4e^{i\theta}$,where $\theta$ is the argument of $\$ 7$

 Step 3: Then, we can rewrite the given expression in terms of $\theta$ and find the real part.

Step 4: To find the maximum real part, we need to analyze the expression and identify theconditions that maximize the real part.

Step 5:We can use trigonometric identities and properties of the real and imaginary parts ofcomplex numbers to simplify the expression and find the maximum real part.

Step 6: By analyzing the expression and using the given conditions, we can determine thevalue of $\theta$ that maximizes the real part and calculate the maximum real part.

 Step 7: Finally, we can provide the answer as the largest possible real part of the givenexpression.

Answer: 1176

Figure 13: Stage 1: Superficial Planning (Initial Step). A baseline response from Qwen 2.5 32B. Themodel generates a "template-like" outline of the solution steps but fails to execute any actual mathematicalderivation, leading to a hallucinated conclusion.

Question:

Find the largest possible real part of $[ ( 7 5 + 1 1 7 \mathrm { i } ) z + ^ { \prime }$ \frac{96+144i}{z}]where $\$ 23$ is a complex number with$\$ 123,45$

Ground Truth: 540

Response: Let's break down and solve the problem step-by-step.

Let $\$ 2$ {i\theta} $=$ 4(cos \theta $+ \ \mathrm { i }$ \sin \theta)$.Substituting this into the expression:  $( 7 5 + 1 1 7 \mathrm { i } )$ \cdot 4 (cos \theta $^ +$ i \sin \theta) $^ +$ \frac{96 + 144i{(4 (cos \theta + i\sin \theta)}]

1. Simplifying the first term:

[4[(75 + 117i)cos \theta $+ \ \mathrm { i }$ \sin \theta) $] = 4$ [(75\cos \theta -117\sin \theta) $+ \ \mathrm { i }$ (75 \sin \theta $^ +$ 117\cos \theta) $] ] [ = 4$ (75\cos \theta-117\sin \theta) $+ \ \mathrm { i }$ \cdot4 (75\sin \theta $^ +$ 117\cos \theta) ]

2.Simplifying the second term:

[\frac{96 + 144i}{4} \cdot (cos \theta -i\sin \theta) = 24(1 + 1.5i)cos \theta -i\sin \theta) $] [ = 2 4$ [cos\theta+ 1.5\sin \theta) $+ \ \mathrm { i }$ (1.5\cos \theta -\sin \theta)]]

3. Combining the real parts:

[\text{Re} = 4(75 \cos \theta -117\sin \theta) + 24(cos \theta + 1.5 \sin \theta)][\text{Re} $= 3 0 0$ \cos\theta -468 \sin \theta $+ ~ 2 4$ \cos \theta +36\sin \theta][\text{Re) $=$ 324\cos \theta -432\sin \theta]

4. Finding the maximum value:

The maximum value of $a \cos \theta $^ +$ b \sin \theta$is $\sqrt{ $\mathord { \uparrow } 2 + \mathsf { b } \mathord { \uparrow } 2 \mathord { \downarrow } \mathfrak { S }$ .For $\$ 4$ and $\$ 6=$-432$:[\text{Max} $=$ \sqrt{324^2 + (-432)^2} = \sqrt{104976 + 186624) $=$ \sqrt{291600}][\sqrt{291600}$= 5 4 0$ 1

Answer: 540

Figure 14: Stage 2: Linear Execution (DAPO Convergence, and FIPO early stage). While the modelcorrectly executes a standard Chain-of-Thought (CoT) to reach the ground truth, its reasoning is strictlylinear and terminates immediately upon finding the first result, lacking any self-verification mechanism.

# Question:

 Find the largest possible real part of $[ ( 7 5 + 1 1 7 \mathrm { i } ) z + ^ { \cdot }$ \frac{96+144i}{z}where $\$ 7$ is a complexnumber with $\$ 123,45$

Ground Truth: 540

# Response:

Let's solve the problem step by step.

We want to find the largest possible real part of the expression. Let $\$ 2$ . Since$\$ | z | =4\ S ,\ S \backslash { \mathsf { f r a c } } \{ 1\} \{ z \} = \backslash { \mathsf { f r a c } } \{ \backslash \mathsf { b a r } \{ z \} \} \{ 1 6 \} \ S$ $\$ 123,45,$

The expression becomes: $( 7 5 + 1 1 7 \mathsf { i } ) z + ( 6 + 9 \mathsf { i } ) \mathsf { \backslash o v e r l i n e \{ z \} . } ]$

Let us write $\$ 23$  in its standard form $\$ 123,45$ . Expanding the terms, the real part of theexpression is $\$ 81 x - 108 y \$ 5$

We need to maximize this subject to the constraint $\$ 123,456$

This is a classic problem for Lagrange multipliers.We want to find $lambda$ such that$\$ 1$ \nabla $\boldsymbol { \mathsf { f } } =$ \lambda\nabla $9 \$ 1$

Solving the system yielded $\$ 1$ $\$ 123,456$ and $y=-\frac{16}{5}$, giving a value of 540.

It seems that at the point $\ S ( \ o \backslash \mathsf { f r a c } \{ 1 2 \} \{ 5 \} , \ \mathsf { - } \backslash \mathsf { f r a c } \{ 1 6 \} \{ 5 \} ) \ S$ we have a value of 540. Hence, themaximum value of the expression appears to be 540.

 Let's confirm if this is indeed the highest value by looking at the geometry of the situation or another approach.

Let us represent the problem in another manner.Let us write $\$ 2$

Then the full expression is expanded in terms of $\$ 123,456$ and $\$ 105,4$

The real part is given by:[324\cos \theta -432\sin \theta.]

To maximize this, we can write it in the form $\$ 1$ \cos(theta $+ \vert \mathsf { p h i } \vert \$ 3$ where $\$ 123,456$$4 3 2 \land 2 \$$

 Let's compute this value step by step. $\$ 324^ { \wedge }2 = 1049 7 6 5\oint \mathrm { 4 } 3 2 ^ { \wedge } 2 = 1 8 6 6 2 4 \oint \mathrm { 8 } \mathrm { R } ^ { \wedge } 2 =$291600 \implies ${ \sf R } = \sf \backslash { s q r t } \{ 2 9 1 6 0 0 \} = 5 4 0 \updownarrow$

Thus, the amplitude R is 540.

This confirms our previous result. Therefore, the largest possible real part of $\$ (75 + 1 1 7 i )z +$\frac $1 9 6 + 1 4 4 \mathrm { i } \{ z \} \$ 5$ is indeed 540.

Answer: 540

Figure 15: Stage 3: Emergent Self-Reflection (FIPO Intermediate Stage). A distinct behavioral shiftoccurs as FIPO training progresses. The model begins to utilize the expanded token budget for sponta-neous self-reflection, cross-validating its initial conclusion through alternative mathematical paths.

Question:

Find the largest possible real part of $[ ( 7 5 + 1 1 7 \mathrm { i } ) z + ^ { \prime }$ \frac{96+144i}{z}]where $z$ is a complex number with$\$ 123,45$

Ground Truth: 540

Response:

Let's solve the given problem step by step.

Let us denote $\$ 23$ as a complex number with $\$ 123,45$ .It implies we can write $z= 4e^{i\theta}$.Then$\frac{1}{z} $=$ \frac{1}{4}e^{-i\theta}$.

 Substituting into the expression:

$( 7 5 + 1 1 7 \mathrm { i } )$ \cdot 4e^{itheta) $^ +$ \frac{96+144i}{4)e^{-i\theta) $=$ (300+468i)e^{i\theta) + (24+36i)e^{-itheta}]Using Euler's formula $e^{i\theta) $=$ \cos\theta $^ +$ isin\theta$,we find the real part:

·From first term: $\$ 300$ \cos\theta -468\sin\theta$.

·From second term: $24\cos\theta $^ +$ 36\sin\theta$.

Total Real Part $= \$ 300+24)$ \cos\theta $^ +$ $- 4 6 8 + 3 6$ )\sin\theta $=$ 324\cos\theta -432\sin\theta$.

The maximum value of $A\cos\theta $^ +$ B\sin\theta$ is $\sqrt{A^2 + B^2}$.

$\mathfrak { S } \mathbb { R } = \mathord { \mathrm { S o r t } } \{ 3 2 4 ^ { \wedge 2 } + ( - 4 3 2 ) ^ { \wedge } 2 \} = \mathord { \mathrm { \backslash } } \mathrm { s q r t } \{ ( 1 0 8 \mathrm { \backslash } \mathrm { c d o t } 3 ) ^ { \wedge } 2 + ( 1 0 8 \mathrm { \backslash } \mathrm { c d o t } 4 ) ^ { \wedge } 2 \} = 1 0 8 \mathord { \mathrm { \backslash } } \mathrm { s q r t } \{ 3 ^ { \wedge } 2 + 4 ^ { \wedge } 2 \} = 1 0 8$\cdot $5 = 5 4 0 \ S$

To verify this reasoning: Let's look at the expression more directly..

let us represent it in the standard form $R\cos(theta+\alpha)$..

[Model re-derives the amplitude formulal]..

This confirms that $\$ 123,405$

To ensure there is no mistake, let me double check my simplification process carefully:

Recall $\$ 123,456$ . Let's recompute it step by step:

· $\$ 324\land2$

· $\$ 432\land2 = 186624$

·$324^2 + 432^2 = 104976 + 186624= 291600$.

$\cdot \ S { \mathsf { R } } = \backslash { \mathsf { s q r t } } \{ 2 9 1 6 0 0 \} = \backslash { \mathsf { s q r t } } \{ ( 5 4 0 ) ^ { \wedge } 2 \} = 5 4 0 \ S .$

[Model then performs a third full-scale re-expansion of the entire complex expression to confirm thecoefficients 324 and -432 are correct.]...

summing up the real components obtained above: $(30\cos\theta - 468\sin\theta) $^ +$ (24\cos\theta $^ +$36\sin\theta)= 324\cos\theta-432\sin\theta$

The result is consistent throughout. Thus, the largest possible real part of $\$ (75 + 1 17 i ) z +\backslash { \mathrm { f r a c } } \{ ( 96 + 14 4 i ) \}$$\{ \boldsymbol { z } \} \$ 5$ with $\$ 123,45$ must indeed then equal $540.$

Answer: 540

Figure 16: Stage 4: Systematic Deep Reasoning (FIPO Late Stage). In the late stages of training, themodel converges on a "compute-heavy" strategy. It moves beyond simple reflection to perform rigorousmulti-pass auditing—including symbolic re-derivation and granular arithmetic verification—to ensurebetter performance in complex reasoning tasks.