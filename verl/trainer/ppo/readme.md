# PPO 改动说明：`use_answer_for_critic` 与 `HEM`

本文总结当前代码中与 **Critic 带标准答案输入** 和 **HEM（双 value 评估）** 相关的修改位置与执行逻辑。

---

## 1. 配置项总览

### 1.1 数据侧（是否给 Critic 注入标准答案）

- `verl/trainer/config/data/legacy_data.yaml:42`
  - `use_answer_for_critic: False`
- `verl/trainer/config/data/legacy_data.yaml:46`
  - `use_solution_for_critic: False`（仅在 `use_answer_for_critic=True` 时生效）
- `verl/trainer/config/data/legacy_data.yaml:53`
  - `standard_solution_key: extra_info.solution`
- `verl/trainer/config/data/legacy_data.yaml:39`
  - `max_prompt_length_with_answer: ${.max_prompt_length}`

### 1.2 HEM 开关与参数

- `verl/trainer/config/algorithm.py:619-624`
  - `use_hem`
  - `hem_alpha`
  - `hem_eps`
  - `hem_clip_min`
  - `hem_clip_max`
  - `pre_update_value_before_adv`
- 默认值在：
  - `verl/trainer/config/ppo_trainer.yaml:123-138`
  - `verl/trainer/config/ppo_megatron_trainer.yaml:141-146`

### 1.3 Critic 双模型路径（HEM）

- `verl/trainer/config/critic/critic.yaml:54,58`
  - `standard_model_path`
  - `oracle_model_path`

### 1.4 BPCO 的 DPPO 与 advantage normalization 开关

- `algorithm.use_dppo=False`
  - 开启后，所有复用 vanilla PPO clipping 的路径改用论文中的 Binary-TV DPPO。
  - `actor_rollout_ref.actor.clip_ratio/clip_ratio_low/clip_ratio_high` 此时表示 sampled token 的绝对概率变化阈值，而不是 PPO ratio 阈值。
- `algorithm.normalize_advantages=True`
  - 设为 `False` 时，GAE/Decoupled-GAE 的 actor advantage 保留原始尺度，不做 batch-wise masked whitening。
  - 该开关不替代 GRPO 专用的 `algorithm.norm_adv_by_std_in_grpo`。

同时启用论文中这两步：

```text
algorithm.use_dppo=True
algorithm.normalize_advantages=False
```

兼容关系：

- HEM：无论 `hem_apply_before_whiten` 为真或假，关闭 normalization 后都保留相同的 HEM 加权顺序，只跳过 whitening。
 - VASPO: clipped probability-ratio reweighting with positive GAE and tanh-gap token selection; see the submission README.
- Rollout correction：普通模式沿用 batch-start `old_log_probs`；若使用 `rollout_correction.bypass_mode=True`，DPPO 的行为策略锚点就是实际 `rollout_log_probs`。

---

## 2. `use_answer_for_critic` / `use_solution_for_critic`：代码位置与逻辑

## 2.1 Dataset 构造 with-answer 输入

- `verl/utils/dataset/rl_dataset.py:113-129`
  - 读取 `max_prompt_length_with_answer`、`use_answer_for_critic`、`use_solution_for_critic`
  - `standard_solution_key` 默认读取 `extra_info.solution`，也兼容顶层 `solution`
- `verl/utils/dataset/rl_dataset.py:265-281`
  - 过滤逻辑按 with-answer 真实 token 长度执行，避免“过滤通过但构造时报 truncation error”
- `verl/utils/dataset/rl_dataset.py:385`
  - `_build_messages_with_answer(...)`
- `verl/utils/dataset/rl_dataset.py:387`
  - 对 `messages` 做 `copy.deepcopy`，避免污染 `raw_prompt`
- `verl/utils/dataset/rl_dataset.py:430-460`
  - 在 user 问题后插入：`The ground truth answer is {answer}.`
  - 开启 solution 后继续插入：`A reference solution:\n{solution}`
- `verl/utils/dataset/rl_dataset.py:501-510`
  - 生成并写入：
    - `input_ids_with_answer`
    - `attention_mask_with_answer`
    - `position_ids_with_answer`

## 2.2 Trainer 中 actor/critic 输入解耦

- `verl/trainer/ppo/ray_trainer.py:1324`
  - `_prepare_actor_critic_inputs(...)`
  - 维护：
    - `*_for_actor`
    - `*_for_critic_standard`
    - `*_for_critic_oracle`
- `verl/trainer/ppo/ray_trainer.py:1400,1409`
  - `_activate_actor_inputs(...)` / `_activate_critic_inputs(...)` 切换输入视图
- 默认路由规则：
  - `use_answer_for_critic=False`：critic 走 standard 输入
  - `use_answer_for_critic=True` 且有 `*_with_answer`：critic 走 oracle 输入
  - `use_solution_for_critic=True` 且同时开启 answer：`*_with_answer` 中的 oracle prompt 为 answer + reference solution
  - 若验证集任一样本缺少非空 solution（包括 schema 有字段但值为 `null`），整个验证集自动退化为 answer-only critic 输入；训练集仍严格要求 solution
  - `use_hem=True`：强制同时保留 standard/oracle 两路，供双 value 评估

示例配置：

```text
data.use_answer_for_critic=True
data.use_solution_for_critic=True
data.standard_answer_key=reward_model.ground_truth
data.standard_solution_key=extra_info.solution
```

---

## 3. HEM：代码位置与逻辑

## 3.1 核心公式

- `verl/trainer/ppo/core_algos.py:327`
  - `compute_hem_weights(...)`
  - 默认 `hem_weight_mode=sequence` 时，使用 `values_standard` 与 `values_oracle` 的 response 区域 masked mean 差值构造 `hem_error`
  - 归一化后得到权重并 clip 到 `[hem_clip_min, hem_clip_max]`
- `algorithm.hem_weight_mode=token`
  - 在 `ray_trainer.py` 中使用每个 valid response token 上的 `|values_oracle-values_standard|` 构造 token-level `hem_error/hem_weight`
  - 若同时设置 `hem_apply_before_whiten=True`，则实现 `masked_whiten(hem_weight * advantage)` 的更新顺序

## 3.2 Trainer 初始化与约束

- `verl/trainer/ppo/ray_trainer.py:400-433`
  - 读取 HEM 参数与模型路径
  - 校验：
    - `use_hem=True` 必须启用 critic
    - 必须 `data.use_answer_for_critic=True`
    - 必须配置 `critic.standard_model_path` 与 `critic.oracle_model_path`

## 3.3 双 value 顺序评估（同一批 response）

- `verl/trainer/ppo/ray_trainer.py:1794`
  - `_compute_value_predictions_for_batch(...)`
  - 顺序：
    1. standard critic + standard 输入 -> `values_standard`
    2. oracle critic + with-answer 输入 -> `values_oracle`
  - 两路都对应同一 `responses`（不改变 actor rollout）

## 3.4 用 HEM 权重缩放 advantage

- `verl/trainer/ppo/ray_trainer.py:1574`
  - `_apply_hem_advantage_weight(...)`
  - `sequence` 模式：`advantages *= hem_weight_clipped.unsqueeze(-1)`
  - `token` 模式：`advantages *= hem_weight_clipped`
  - 同时记录 `hem_error/weight` 与统计指标

## 3.5 Critic 更新（standard + oracle）

- `verl/trainer/ppo/ray_trainer.py:1939`
  - `_run_critic_update_step(...)`
  - 先更新 standard critic；若 `use_hem=True` 再切 oracle 输入更新 oracle critic
  - oracle 指标重命名到 `critic_oracle/*`

## 3.6 HEM 模型路径处理与 HF 转换

- `verl/trainer/ppo/ray_trainer.py:1518`
  - `_ensure_hem_model_paths_ready(...)`
  - 支持 sharded checkpoint 直载；非 sharded（如 HF 目录）会转换为 runtime sharded ckpt
- `verl/workers/fsdp_workers.py:1566,1649`
- `verl/workers/megatron_workers.py:1249,1330`
  - 提供 `reinit_model_from_path(...)` 与 `clear_checkpoint_history(...)` 供 trainer 复用

## 3.7 pre-update value 再算 GAE

- `verl/trainer/ppo/ray_trainer.py:2294,2342-2348`
  - 当 `pre_update_value_before_adv=True`：
    1. 先用当前 batch 更新一次 critic
    2. 再重新跑 value 评估
    3. 再重新计算 advantage/GAE

---

## 4. 监控指标（与本次改动相关）

## 4.1 HEM 基础统计

- `verl/trainer/ppo/ray_trainer.py:1574`
  - `hem_error_mean/std`
  - `hem_weight_mean/min/max`
  - `value_standard_mean`
  - `value_oracle_mean`

## 4.2 Value 分段监控（正确/错误 + 位置分桶）

- `verl/trainer/ppo/ray_trainer.py:1657`
  - standard：`critic/value_monitor/...`
  - oracle：`critic_oracle/value_monitor/...`
  - 分桶：`first_1_3, middle_1_3, last_1_3, last5`
- `verl/trainer/ppo/ray_trainer.py:1687`
  - 新增 `critic_oracle/vf_explained_var`
  - 新增 standard/oracle 的分段 explained variance：`vf_explained_var/first_1_3`,
    `vf_explained_var/middle_1_3`, `vf_explained_var/last_1_3`

## 4.3 Advantage 分段监控（同分桶逻辑）

- `verl/trainer/ppo/ray_trainer.py:1693`
  - `actor/adv_monitor/...`

## 4.4 Value 路径挑选 Accuracy

- `train_value_model/...` 和 `train_value_model_oracle/...`
  - 分段方法改为：`first_1_3`, `middle_1_3`, `last_1_3`
  - 额外保留：`last_5_tokens`
  - 全回复均值仍保留：`ratio_0_1`

## 4.5 每步打印 HEM 极值样本回复

- `verl/trainer/ppo/ray_trainer.py:1614`
  - `_log_hem_extreme_responses(...)`
  - 打印当步 `hem_error` 最大/最小样本的 response 文本
- 调用点：`verl/trainer/ppo/ray_trainer.py:2366`

---

## 5. 行为退化（兼容性）

- `use_answer_for_critic=False`
  - Critic 路径退化为原始 prompt 输入，不依赖 `*_with_answer`
- `use_hem=False`
  - 不执行第二次 oracle value 评估
  - 不执行 HEM 加权
  - 不更新 oracle critic 分支
- `pre_update_value_before_adv=False`
  - 保持原 PPO 时序（先 value/adv，再 update critic）

---

## 6. 额外相关项（可选）

- `use_value_sigmoid`（value 输出 sigmoid 到 `[0,1]`）
  - 配置：`verl/trainer/config/critic/critic.yaml:62`
  - 生效位置：
    - `verl/workers/critic/dp_critic.py:132`
    - `verl/workers/critic/megatron_critic.py:129,230`
    - `verl/workers/utils/losses.py:190`
    - legacy infer 路径：`verl/trainer/ppo/ray_trainer.py:1786`
