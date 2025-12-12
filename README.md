# Routing and Spectrum Allocation using Deep Q-Learning

This project implements a custom Gymnasium environment for the Routing and Spectrum Allocation (RSA) problem and trains a Deep Q-Network (DQN) agent to minimize the blocking probability in an elastic optical network. The full pipeline includes: environment design, state transitions, action representation, reward structure, training configuration, hyperparameter tuning, and evaluation on unseen request files.

### How to Execute

1. Clone Repo
```
git clone https://github.com/Mandar219/rsa.git
cd rsa
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Train the DQN agent
```
python dqn_runner.py --train_dir data/train --eval_dir data/eval
```

This will:
- Train two agents (capacity 10 and 20)
- Save models under `models/`
- Save training curves and evaluation blocking plots under `results/`

All results are fully reproducible because:
- Training, evaluation, and environment state transitions are deterministic except for exploration
-  Request files are fixed CSV inputs
- The deterministic=True mode is used during evaluation

### Environment

Our environment follows the Gymnasium Env API, including:

- `reset() → (state, info)`

- `step(action) → (state, reward, terminated, truncated, info)`

Environment file: `rsaenv.py`

**Topology**

A fixed 9-node graph with:

- A 9-node ring
- Three shortcut edges

Each undirected link carries a fixed number of wavelengths (capacity ∈ {10, 20}).

**Episode Structure**
- One episode = one request file (100 requests)
- At each time step:

    1. A new request arrives
    2. The agent chooses a path
    3. The environment checks wavelength availability
    4. A wavelength is allocated or blocked
    5. All link holding times decrement by 1
    6. Next observation is returned

### State Representation and State Transitions

**State vector**

The observation is a single flattened vector consisting of:

1. Link spectra:
For each link, we store its per-wavelength remaining holding time, normalized to [0,1].

2. Current request:

    `[normalized_src, normalized_dst, normalized_holding_time]`

Total dimension:
```nginx
num_links * capacity + 3
```

**State Transition**

At every request arrival (one time slot):

1. Agent selects a path (one of 8 predefined candidate paths)

2. Environment checks wavelength continuity:

    - Same wavelength index must be free on all links in the path

3. If free:

    - Allocate wavelength w

    - Set holding time h

4. After processing:

    - Decrement all positive wavelength timers on all links

This ensures that ongoing lightpaths occupy wavelengths for their entire holding time, as in real RSA systems.

**LinkState data structure**

Defined in `nwutil.py`:
```python
class LinkState:
    self.spectrum = np.zeros(capacity)
```

Where:

- spectrum[w] = remaining holding time
- 0 means free
- advance_time() decrements non-zero entries

This allows O(1) wavelength checks and updates.

### Action Representation

Each action corresponds to a specific candidate path among the eight predefined routes (P1–P8). These were taken directly from the project specification.

Action space:
```python
spaces.Discrete(8)
```

Actions invalid for a given (src,dst) pair automatically yield a blocked request, allowing the agent to learn the valid routing choices.

### Reward Function

The reward is:
```python
+1  if request allocated successfully
−1  if request is blocked
```

This aligns reward maximization with minimizing the blocking probability, our objective.

**Additional Constraints**
- Wavelength continuity constraint enforced across all links in the chosen path.
- First-fit wavelength assignment: allocate the smallest free wavelength index.
- No wavelength conversion allowed.
- LinkState timers ensure correct teardown of circuits.

### Training Setup

**DQN configuration (stable-baselines3)**

```python
DQN(
  policy="MlpPolicy",
  learning_rate=1e-3,
  buffer_size=50_000,
  batch_size=64,
  gamma=0.99,
  train_freq=1,
  gradient_steps=1,
  exploration_fraction=0.3,
  exploration_final_eps=0.05,
  target_update_interval=1000
)
```

Training for each capacity setting:
- 200,000 timesteps
- Random train-file selection each episode
- Deterministic evaluation

**Hyperparameter Tuning**

Our tuning process focused on a small set of practical adjustments to identify stable hyperparameters for DQN under the RSA environment.

We performed lightweight manual tuning rather than an exhaustive grid search. We adjusted a few key parameters and selected values that provided smooth learning curves and decreasing blocking probability.

1. Learning Rate

    We compared 1e-3 and 5e-4.

    1e-3 converged faster and produced less noisy learning curves, so we selected it.

    `dqn_runner.py`
    ```python
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-3,
        ...
    )
    ```

2. Batch Size

    We tested batch size of 32 and 64.

    Batch size of 64 produced more stable updates.

    ```python
    batch_size=64,
    ```

3. Exploration Schedule

    We compared:

    - exploration_fraction=0.3, exploration_final_eps=0.05

    - A smaller exploration schedule (0.2 → 0.02)

    The smaller exploration collapsed too early.
    The configuration 0.3 → 0.05 allowed sufficient exploration during early training and converged reliably.

    ```python
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    ```

4. Replay Buffer Size

    We tried buffer size of 25,000 and 50,000.

    The buffer size of 50,000 produced smoother learning curves and better stability.

    ```python
    buffer_size=50_000,
    ```

The final chosen hyperparameters correspond to the configuration shown in the `DQN(...)` block of `dqn_runner.py`.

### Results

Below are the required six plots with a brief explanation each.

1. Learning Curve (Capacity = 10)
![](/results/capacity_10/learning_curve_cap10.png)

Shows averaged episode returns. Performance improves from negative returns to ~80+.

2. Objective B During Training (Capacity = 10)
![](/results/capacity_10/training_blocking_cap10.png)

Blocking probability decreases from ~0.75 to ~0.10 as agent learns better paths.

3. Evaluation Blocking Curve (Capacity = 10)
![](/results/capacity_10/eval_blocking_cap10.png)

Evaluated with deterministic policy on unseen files. Blocking stabilizes around ~0.05–0.14.

4. Learning Curve (Capacity = 20)
![](/results/capacity_20/learning_curve_cap20.png)

Faster convergence and higher returns than capacity=10 due to more wavelengths.

5. Objective B During Training (Capacity = 20)
![](/results/capacity_20/training_blocking_cap20.png)

Blocking probability decreases from ~0.75 to ~0.05.

6. Evaluation Blocking Curve (Capacity = 20)
![](/results/capacity_20/eval_blocking_cap20.png)

Evaluated on unseen files. Blocking is very low (~0–0.05).

### Source Code Files

`rsaenv.py` — Custom RSA Gymnasium Environment

Implements:
- Topology construction
- State transitions
- Wavelength allocation
- Reward logic
- Blocking-rate reporting

`dqn_runner.py` — Agent Training Pipeline

Includes:

- Environment creation
- DQN training
- Logging 
- Saving models and plots
- Deterministic evaluation loop

`nwutil.py` — Network Utility Functions

Includes:

- LinkState class
- Topology definition
- Path definitions
- CSV request loader 

### Trained Models

Trained models for capacity 10 and 20 are saved as:

```
models/capacity_10/dqn_capacity_10.zip
models/capacity_20/dqn_capacity_20.zip
```

These can be loaded via:

```python
from stable_baselines3 import DQN
model = DQN.load("models/capacity_10/dqn_capacity_10.zip")
```

### Summary

This project demonstrates:

- A fully functional RSA environment with wavelength continuity
- A DQN agent that effectively reduces blocking probability
- Proper training methodology and reproducible evaluation
- Strong performance generalization on unseen request sequences

The agent consistently learns near-optimal path-selection policies under realistic network constraints.