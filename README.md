# Cricket Win Predictor (T20)

A machine learning project to model and predict win probabilities during T20 cricket matches using ball-by-ball match data from Cricsheet.

---

## 1. Purpose of the Project

The purpose of this project is to build an accurate, data-driven Cricket Win Predictor for Twenty20 (T20) matches. By analyzing ball-by-ball match dynamics—including run rates, wickets fallen, balls remaining, match conditions, and pitch/venue context—the model estimates the live win probability for both teams throughout an innings.

---

## 2. Raw Data Storage (`data/raw/`)

All original, raw Cricsheet JSON match files are placed in `data/raw/`.

- **Immutability**: Raw data files are treated as read-only single sources of truth.
- **Reproducibility**: Keeping raw inputs untouched guarantees that downstream data pipelines and feature extraction steps can be reproduced from scratch at any time.

---

## 3. Processed Data Storage (`data/processed/`)

Transformed and structured tabular outputs (such as `data/processed/ball_by_ball.csv` and `data/processed/match_state.csv`) are written into `data/processed/`.

- **Pipeline Separation**: Keeps transformed datasets cleanly decoupled from original data.
- **Performance**: Provides fast loading for exploratory analysis, feature engineering, and model training.

---

## 4. What One Row Represents (Phase 1)

In the raw delivery dataset (`ball_by_ball.csv`):

- **One row = One delivery recorded in the Cricsheet match data.**
- Every single delivery event (legal deliveries, wides, no-balls, penalty runs, and wickets) is captured as a distinct row reflecting the state of play at that instant.

---

## 5. How Legal Balls Are Calculated

Cricket rules dictate that an over consists of 6 **legal** deliveries. To accurately compute legal balls:

1. For each innings, `legal_ball` count begins at `0`.
2. A standard legal delivery increments `legal_ball` by `1`.
3. An extra delivery resulting from a **wide** or **no-ball** does **not** increment `legal_ball`.
4. Other deliveries (including byes and leg-byes, which are legal deliveries off the bat/body) increment `legal_ball`.
5. The `ball_in_over` counter tracks the legal ball sequence within the current over (1 to 6) and resets to 0 whenever a new over commences.

Example:
- Delivery 1 (legal): `legal_ball` = 1, `ball_in_over` = 1
- Delivery 2 (legal): `legal_ball` = 2, `ball_in_over` = 2
- Delivery 3 (wide): `legal_ball` = 2, `ball_in_over` = 2 (not incremented)
- Delivery 4 (legal): `legal_ball` = 3, `ball_in_over` = 3

---

## 6. Why Wides and No-Balls Require Special Handling

- In Cricsheet data, deliveries are listed in sequential order, but nominal delivery indices do not distinguish between legal balls and illegitimate deliveries (wides/no-balls).
- In a match, a bowler must re-bowl any wide or no-ball; thus an over may contain 7, 8, or more actual deliveries to achieve 6 legal balls.
- Win prediction models rely heavily on **balls remaining** (out of 120 legal balls in a standard T20 innings) and **required run rate**. Calculating these metrics based on nominal delivery count rather than verified legal deliveries would distort ball counts, run rates, and probability estimates.

---

## 7. How to Run the Phase 1 Parser

Once raw Cricsheet JSON files are uploaded to `data/raw/`, run the parser via the command line:

```bash
# Parse a single match JSON file
python src/data/parse_cricsheet.py --input data/raw/1001349.json

# Parse all match JSON files in the raw directory
python src/data/parse_cricsheet.py --input data/raw/
```

Parsed outputs will be saved to `data/processed/ball_by_ball.csv`.

---

## 8. Phase 2: Match-State Feature Engineering

Phase 2 converts the raw delivery data into a clean, leakage-free **match-state dataset** (`data/processed/match_state.csv`) specifically engineered to answer:

> *"Given the current state of a T20 chase, what is the probability that the chasing team will eventually win?"*

### 8.1 What a Match-State Row Represents
Each row represents the instantaneous state of a 2nd innings chase immediately **AFTER** that delivery event. It captures:
- Current score and wickets lost at that precise instant.
- Target score and runs remaining to win.
- Legal balls remaining and overs completed.
- Current run rate and required run rate.
- Ground truth target label: `chasing_team_won` (0 or 1).

### 8.2 Why Only Second Innings Are Used
The baseline win probability model models a run-chase scenario. In the second innings, the target score is fixed, allowing exact mathematical calculation of `runs_remaining` and `required_run_rate`. The first innings represents par-score setting rather than chasing, which involves different game dynamics.

### 8.3 Target Score Calculation
$$\text{target\_score} = \text{final\_score}_{\text{innings 1}} + 1$$
Example: If Team 1 scores 180 runs, `target_score` is 181.

### 8.4 Runs Remaining Calculation
$$\text{runs\_remaining} = \max(0, \text{target\_score} - \text{current\_score})$$
If the chasing team reaches or exceeds the target, `runs_remaining` cleanly clamps to `0`.

### 8.5 Balls Remaining Calculation
A standard T20 innings contains 120 legal balls (20 overs $\times$ 6 legal deliveries). Based on actual legal deliveries completed:
$$\text{balls\_remaining} = \max(0, 120 - \text{legal\_balls\_completed})$$
Wides and no-balls do not advance `legal_balls_completed` and therefore do not reduce `balls_remaining`.

### 8.6 Current Run Rate Calculation
$$\text{overs\_completed} = \frac{\text{legal\_balls\_completed}}{6.0}$$
$$\text{current\_run\_rate} = \begin{cases} \frac{\text{current\_score}}{\text{overs\_completed}} & \text{if } \text{legal\_balls\_completed} > 0 \\ 0.0 & \text{if } \text{legal\_balls\_completed} = 0 \end{cases}$$
Zero division is prevented when 0 legal deliveries have occurred.

### 8.7 Required Run Rate Calculation
$$\text{overs\_remaining} = \frac{\text{balls\_remaining}}{6.0}$$
$$\text{required\_run\_rate} = \begin{cases} \frac{\text{runs\_remaining}}{\text{overs\_remaining}} & \text{if } \text{balls\_remaining} > 0 \text{ and } \text{runs\_remaining} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
Edge cases (balls exhausted or target achieved) safely default to `0.0`, eliminating infinite or NaN values.

### 8.8 Chasing Team Won Label (`chasing_team_won`)
- `1` = Chasing team (`batting_team` in innings 2) eventually won the match.
- `0` = Chasing team eventually lost the match (`winner == bowling_team`).

### 8.9 Strict Data Leakage Prevention
To guarantee valid machine learning modeling:
- The `winner` column is strictly excluded from `match_state.csv`.
- Match margins, final innings 2 totals, and future deliveries/wickets are excluded.
- Every feature is purely backward-looking and known at the delivery moment.

### 8.10 Match Inclusions and Exclusions
Out of **5,729** total matches in Cricsheet raw data:
- **Included**: **5,563 matches** (97.10%)
- **Excluded**: **166 matches** (2.90%)
  - *84 matches*: No second innings took place (abandoned due to weather or awarded prior to chase).
  - *82 matches*: No definitive winner determined (ties without super-over winner, or rain abandonments mid-chase).

### 8.11 How to Run the Phase 2 Feature Pipeline
```bash
python src/features/build_features.py --input data/processed/ball_by_ball.csv --output data/processed/match_state.csv
```
Output: `data/processed/match_state.csv` (588,959 rows $\times$ 13 columns).

---

## 9. Phase 3: Baseline Win-Probability Model

Phase 3 builds the first baseline predictive model for T20 run chases using **Logistic Regression**.

### 9.1 Why Logistic Regression is the Baseline
- **Interpretability**: Linear combination of standardized features mapped through the logit link function provides transparent, inspectable coefficients.
- **Probabilistic Output**: Directly outputs continuous probabilities $P(\text{chase win}) \in [0, 1]$ via the sigmoid function.
- **Benchmark Standard**: Fast and reproducible baseline against which more complex models (Random Forests, Gradient Boosting) can be objectively compared.

### 9.2 Match-Level Train/Test Split (Preventing Data Leakage)
A single cricket match generates between 60 to 130 sequential delivery rows. Splitting deliveries randomly across rows would cause severe intra-match leakage (rows from the same match sharing identical pitch conditions, weather, team lineups, and target score appearing in both train and test).
- **Split Strategy**: 80% unique matches for training (4,450 matches, 471,169 rows), 20% held-out matches for testing (1,113 matches, 117,790 rows).
- **Reproducibility**: Fixed `random_state = 42`.
- **Overlap**: Exactly **0** matches overlap between train and test sets.

### 9.3 Feature Set Used
Eight purely numerical match-state features:
1. `target_score`
2. `current_score`
3. `wickets_lost`
4. `runs_remaining`
5. `balls_remaining`
6. `overs_completed`
7. `current_run_rate`
8. `required_run_rate`

### 9.4 Target Variable
`chasing_team_won`:
- `1` if chasing team won
- `0` if defending/bowling team won

### 9.5 Why Probability is More Important Than Classification
Cricket analysts, viewers, and teams do not just care about a static binary prediction ("Win" or "Loss"). They require continuous tracking of dynamic momentum swings (e.g., from 40% to 75% after a 20-run over). The quality of probability estimates across all gamestates matters far more than simple 50% threshold accuracy.

### 9.6 Model Evaluation Metrics
Evaluated exclusively on the **117,790 held-out test deliveries** across 1,113 unobserved matches:
- **ROC-AUC**: **0.9109** (Demonstrates exceptional ranking discrimination).
- **Accuracy**: **82.22%**
- **Precision**: **80.15%**
- **Recall**: **79.33%**
- **F1 Score**: **79.74%**
- **Brier Score**: **0.1204** (Mean squared difference between predicted probabilities and actual binary outcomes; lower is better, representing strong calibration).

### 9.7 Probability Calibration
Calibration assesses whether a predicted $X\%$ probability translates into an actual $X\%$ historical win rate. The generated reliability curve ([outputs/calibration_curve.png](file:///c:/Users/HARIHARAN/Desktop/cric/outputs/calibration_curve.png)) closely tracks the diagonal 45-degree reference line, confirming reliable probability predictions across deciles.

### 9.8 Feature Coefficients & Interpretability
Standardized Logistic Regression coefficients (from [outputs/logistic_coefficients.csv](file:///c:/Users/HARIHARAN/Desktop/cric/outputs/logistic_coefficients.csv)):
- `current_score` (+1.1849): Positive association with chase win probability.
- `balls_remaining` (+0.4224): More balls in hand increases chase probability.
- `current_run_rate` (+0.2700): Higher existing momentum positively correlates with success.
- `target_score` (-0.0820): Higher initial targets reduce chase probability.
- `overs_completed` (-0.4226): More overs expended holding score constant lowers win probability.
- `runs_remaining` (-1.0027): Larger run deficits reduce win probability.
- `wickets_lost` (-1.0948): Loss of wickets severely harms chase probability.
- `required_run_rate` (-8.6677): Highest negative influence; steep required rates rapidly diminish chase probability.

*(Note: Coefficients denote directional associations holding other standardized model features constant, not causal claims).*

### 9.9 Saved Model Artifact
The model is saved as an end-to-end `sklearn.pipeline.Pipeline` with `StandardScaler` and `LogisticRegression` using `joblib`:
- File: `models/logistic_regression_win_probability.joblib`
- Features are scaled internally using parameters learned exclusively from training data.

### 9.10 How to Make Predictions
Use the standalone inference script [src/models/predict.py](file:///c:/Users/HARIHARAN/Desktop/cric/src/models/predict.py):

```bash
python src/models/predict.py \
  --target_score 180 \
  --current_score 120 \
  --wickets_lost 4 \
  --runs_remaining 60 \
  --balls_remaining 36 \
  --overs_completed 14.0 \
  --current_run_rate 8.57 \
  --required_run_rate 10.0
```

---

## 10. Phase 4: Model Comparison — Logistic Regression vs Random Forest

Phase 4 evaluates whether an ensemble tree-based model (**Random Forest**) improves upon the linear baseline in discrimination, classification, and probability calibration.

### 10.1 Why Random Forest was Compared
- **Non-Linear Partitions**: Can learn non-linear boundaries (e.g., 30 runs needed off 6 balls vs 30 off 18 balls).
- **Feature Interactions**: Automatically models complex multi-way interactions (e.g. required run rate interacting with wickets in hand) without explicit manual interaction terms.
- **Ensemble Robustness**: Averaging across 300 decision trees reduces variance.

### 10.2 Strictly Reproducible Match Split
To ensure a completely fair comparison, Random Forest was evaluated on the exact same train/test partition as Phase 3:
- **Training Matches**: 4,450 (471,169 rows)
- **Testing Matches**: 1,113 (117,790 rows)
- **Split Seed**: `random_state = 42` (Zero match overlap).

### 10.3 Preprocessing Differences
- **Logistic Regression**: Requires `StandardScaler` to place all features on comparable numerical scales for gradient optimization and coefficient regularization.
- **Random Forest**: Invariant to monotonic scale transformations. Features are fed directly in raw numerical form.

### 10.4 Performance Comparison Table

| Metric | Logistic Regression | Random Forest (300 Trees) | Difference ($\Delta$) |
|---|---|---|---|
| **Accuracy** | **82.22%** (0.8222) | 82.11% (0.8211) | -0.11% |
| **Precision** | **80.15%** (0.8015) | 80.02% (0.8002) | -0.13% |
| **Recall** | **79.33%** (0.7933) | 79.20% (0.7920) | -0.13% |
| **F1 Score** | **79.74%** (0.7974) | 79.61% (0.7961) | -0.13% |
| **ROC-AUC** | 0.9109 | **0.9113** | **+0.0004** |
| **Brier Score** | **0.1204** | 0.1209 | +0.0005 (LR is slightly better calibrated) |

### 10.5 Probability Quality & Calibration Insights
- **ROC-AUC**: Random Forest achieves a marginally higher ROC-AUC (0.9113 vs 0.9109), showing slightly superior ranking discrimination.
- **Brier Score & Calibration**: Logistic Regression achieves a marginally superior Brier Score (0.1204 vs 0.1209). Because Logistic Regression minimizes cross-entropy loss directly, its raw sigmoid probabilities are better calibrated across probability extremes than uncalibrated tree-leaf frequency averaging.
- **Curves**: Both curves are visually compared in [outputs/model_comparison_roc_curve.png](file:///c:/Users/HARIHARAN/Desktop/cric/outputs/model_comparison_roc_curve.png) and [outputs/model_comparison_calibration_curve.png](file:///c:/Users/HARIHARAN/Desktop/cric/outputs/model_comparison_calibration_curve.png).

### 10.6 Random Forest Feature Importance
Model-based Gini impurity reduction from [outputs/random_forest_feature_importance.csv](file:///c:/Users/HARIHARAN/Desktop/cric/outputs/random_forest_feature_importance.csv):
1. **`required_run_rate`**: **38.57%** (Primary tree partition feature)
2. **`target_score`**: **16.67%**
3. **`runs_remaining`**: **12.93%**
4. **`current_run_rate`**: **11.51%**
5. **`wickets_lost`**: **10.18%**
6. **`current_score`**: **4.07%**
7. **`overs_completed`**: **3.10%**
8. **`balls_remaining`**: **2.97%**

*(Note: Feature importance represents model split frequency and variance reduction, NOT real-world causal impact).*

### 10.7 Model Agreement & Disagreement Analysis
- **Agreement**: The models agree closely across the vast majority of delivery states:
  - Median absolute probability difference: **4.00%**
  - Mean absolute probability difference: **6.94%**
  - **75.8%** of all test delivery predictions agree within 10 percentage points.
- **Key Disagreement Scenarios**: The largest disagreements occur at boundary conditions when balls are exhausted (`balls_remaining = 0` with `runs_remaining > 0`):
  - Random Forest splits directly on `balls_remaining <= 0` and correctly predicts near-zero win probability (0.5%–0.8%).
  - Logistic Regression, without non-linear interaction terms and receiving `required_run_rate = 0.0`, is susceptible to edge-case extrapolation.

### 10.8 How to Run Model Comparison
```bash
python src/models/compare_models.py
```
Outputs are saved in `outputs/` and `models/random_forest_win_probability.joblib`.

---

## 11. Phase 5: Win Probability Swings & Match Moment Analysis

Phase 5 calculates delivery-by-delivery win probability changes using the trained baseline model to quantify the statistical impact of specific game events.

### 11.1 What Win Probability Represents
Win probability $P(\text{chase win} \mid \text{state}_t)$ is the estimated statistical likelihood that the chasing team will eventually reach the target, conditioned purely on the current state (score, wickets lost, balls remaining, run rates) at delivery $t$.

### 11.2 How Probability Change is Calculated
For each delivery $t$ within a match:
$$\Delta P_t = P_t - P_{t-1}$$
$$\text{Absolute Swing}_t = |\Delta P_t|$$
- Swings are calculated **strictly within each match independently**. No probability comparisons cross match boundaries.
- For the initial delivery of each second innings ($t=1$), $P_{t-1} = \text{NaN}$, yielding $\Delta P_1 = \text{NaN}$.

### 11.3 What Positive and Negative Swings Mean
- **Positive Swing ($\Delta P > 0$)**: The delivery event increased the chasing team's estimated chance of winning (e.g. boundary hit, extras conceded, high-scoring over).
- **Negative Swing ($\Delta P < 0$)**: The delivery event decreased the chasing team's estimated chance of winning (e.g. wicket taken, dot ball in death overs, rising required rate).

### 11.4 Why "Win Probability Swing" is Used Instead of "Momentum"
"Momentum" is a colloquial and psychological term frequently assumed to be a latent physical or psychological force. In predictive analytics, we avoid claiming "momentum" because probability shifts simply measure changes in the underlying objective match equation (balls remaining vs runs required vs wickets in hand). We refer to these strictly as **"Win Probability Swings"** or **"Win Probability Changes"**.

### 11.5 Terminal-State Handling
When a chase reaches its definitive conclusion:
- **Case A**: `runs_remaining <= 0` (Target achieved) $\rightarrow$ Win probability is deterministic: **1.0 (100%)**.
- **Case B**: `balls_remaining <= 0` and `runs_remaining > 0` (Deliveries exhausted without reaching target) $\rightarrow$ Win probability is deterministic: **0.0 (0%)**.
Applying this domain-grounded terminal correction ensures that model boundary extrapolation on final balls does not create artificial or distorted swings.

### 11.6 Strict Leakage Prevention
The Logistic Regression model generates predictions using **only the 8 backward-looking match state features**. The match outcome (`winner`) is never passed into the model feature vector and is only referenced to assign the deterministic terminal state after match conclusion.

### 11.7 Cricket Events and Associated Swings
Summary across 583,396 valid delivery swings:

| Event Category | Deliveries | Mean $\Delta P$ | Median $\Delta P$ | Largest Drop | Largest Gain |
|---|---|---|---|---|---|
| **Wicket** | 32,654 | **-0.0594** (-5.94%) | -0.0458 (-4.58%) | **-0.9736** (-97.36%) | +0.5353 (+53.53%) |
| **Boundary (4/6)** | 69,901 | **+0.0527** (+5.27%) | +0.0422 (+4.22%) | -0.9955 (-99.55%) | **+1.0000** (+100.0%) |
| **Dot Ball** | 218,579 | **-0.0124** (-1.24%) | -0.0097 (-0.97%) | -0.7255 (-72.55%) | +0.6529 (+65.29%) |
| **Other Scoring (1,2,3,5)** | 262,262 | **+0.0050** (+0.50%) | +0.0013 (+0.13%) | -0.5366 (-53.66%) | +0.9983 (+99.83%) |

*(Note: Values represent statistical associations with event occurrences, not independent causal claims).*

### 11.8 Visualizations
- **Swing Distribution**: [outputs/swing_distribution.png](file:///c:/Users/HARIHARAN/Desktop/cric/outputs/swing_distribution.png) shows the density of delivery-level probability shifts centered around 0 with heavy tails corresponding to key wickets and death-over boundaries.
- **Match Timeline**: [outputs/match_probability_example.png](file:///c:/Users/HARIHARAN/Desktop/cric/outputs/match_probability_example.png) charts the complete 120-delivery probability progression for Match 430885 (India vs Sri Lanka, target 207), highlighting turning points.

### 11.9 How to Run Phase 5 Analysis
```bash
python src/analysis/probability_swings.py
```
Outputs:
- `outputs/win_probability_swings.csv` (complete 588,959-row dataset with swings)
- `outputs/top_positive_swings.csv` (top 100 positive swings)
- `outputs/top_negative_swings.csv` (top 100 negative swings)
- `outputs/swing_event_summary.csv`
- `outputs/top_swing_matches.csv`

---

## 12. Phase 6: What-If Next-Over Win Probability Simulator

Phase 6 implements a scenario-based what-if simulator that evaluates how different hypothetical next-over outcomes (runs and wickets) would alter the chasing team's estimated win probability.

### 12.1 Purpose of the Simulator
The simulator is an **evaluative scenario tool**, not a next-over outcome predictor. It allows users, coaches, analysts, and fans to inspect the sensitivity of the match equation:
- *"If the bowling side restricts the chasing team to a dot over, how much does the win probability decline?"*
- *"If the batting side takes 12 runs without losing a wicket, does it tilt the odds in their favor?"*

### 12.2 How Match State is Updated After a Hypothetical Over
Given a current match state, an over consisting of 6 legal deliveries is simulated by applying incremental hypothetical events $(\Delta \text{runs}, \Delta \text{wickets})$:
- $\text{Score}_{\text{new}} = \text{Score}_{\text{current}} + \Delta \text{runs}$
- $\text{Wickets}_{\text{new}} = \min(10, \text{Wickets}_{\text{current}} + \Delta \text{wickets})$
- $\text{Balls Remaining}_{\text{new}} = \max(0, \text{Balls Remaining}_{\text{current}} - 6)$
- $\text{Overs Completed}_{\text{new}} = \text{Overs Completed}_{\text{current}} + 1.0$ (or fractional if fewer than 6 balls remain)
- $\text{Runs Remaining}_{\text{new}} = \max(0, \text{Target} - \text{Score}_{\text{new}})$
- $\text{Current Run Rate (CRR)}_{\text{new}} = \frac{\text{Score}_{\text{new}}}{\text{Overs Completed}_{\text{new}}}$
- $\text{Required Run Rate (RRR)}_{\text{new}} = \frac{\text{Runs Remaining}_{\text{new}}}{\text{Balls Remaining}_{\text{new}} / 6}$ (or $0.0$ if runs or balls $\le 0$)

### 12.3 Terminal-State Handling
Cricket matches end immediately upon hitting deterministic boundaries. To prevent model extrapolation artifacts, the simulator enforces exact cricket domain rules:
- **Target Reached**: If $\text{Score}_{\text{new}} \ge \text{Target}$, win probability is set to **1.0 (100%)**.
- **Balls Exhausted**: If $\text{Balls Remaining}_{\text{new}} \le 0$ and $\text{Score}_{\text{new}} < \text{Target}$, win probability is set to **0.0 (0%)**.
- **All Out**: If $\text{Wickets}_{\text{new}} \ge 10$ and $\text{Score}_{\text{new}} < \text{Target}$, win probability is set to **0.0 (0%)**.

### 12.4 Why Simulation Does NOT Predict Actual Next-Over Outcomes
The simulator does **not** estimate the probability of scoring 6, 10, or 18 runs, nor does it forecast bowler tactics or batter execution. Instead, it tests the **model's response function** conditional on specified hypothetical inputs. It answers: *"Assuming outcome X happens, what would the resulting win probability be?"*

### 12.5 Why This Does Not Introduce Target Leakage
The simulation pipeline is strictly feed-forward:
1. It takes the user-supplied or historical state.
2. It projects an arithmetic match state for the end of the over.
3. It passes this synthesized state through the frozen, pre-trained Phase 3 Logistic Regression pipeline.
At no point is the true match outcome or future delivery log accessed or leaked.

### 12.6 Example Interpretation (TEST A: 142/4 chasing 190, 24 balls left)
- **Current State**: Target 190, 142/4 in 16.0 overs. Equation: Need 48 off 24 balls ($\text{CRR} = 8.88$, $\text{RRR} = 12.00$). Current win probability is **40.6%**.
- **Impact of a Dot Over (0 runs, 0 wkts)**: The equation worsens to 48 off 18 balls ($\text{RRR} = 16.00$), plunging win probability by **-30.6%** to **10.0%**.
- **Impact of an Explosive Over (18 runs, 0 wkts)**: The equation eases to 30 off 18 balls ($\text{RRR} = 10.00$), surging win probability by **+36.4%** to **77.0%**.
- **Wicket Penalty**: Comparing 10 runs with 0 wickets (+1.8%, probability 42.4%) versus 10 runs with 1 wicket (-8.9%, probability 31.7%) reveals that a wicket in this phase of the chase costs approximately **10.7 percentage points** in win expectancy.

### 12.7 Summary Table of Controlled Scenarios (TEST A)

| Scenario | Runs | Wkts | New Score | Win Prob | Change | Terminal State |
|---|---|---|---|---|---|---|
| **Dot Over** | 0 | 0 | 142/4 | **10.0%** | **-30.6%** | Non-Terminal |
| **Low Scoring Over** | 4 | 0 | 146/4 | **19.1%** | **-21.5%** | Non-Terminal |
| **Moderate Over** | 6 | 0 | 148/4 | **25.6%** | **-15.0%** | Non-Terminal |
| **Good Over** | 8 | 0 | 150/4 | **33.5%** | **-7.1%** | Non-Terminal |
| **Strong Over** | 10 | 0 | 152/4 | **42.4%** | **+1.8%** | Non-Terminal |
| **Very Good Over** | 12 | 0 | 154/4 | **51.8%** | **+11.2%** | Non-Terminal |
| **Excellent Over** | 15 | 0 | 157/4 | **65.5%** | **+24.9%** | Non-Terminal |
| **Explosive Over** | 18 | 0 | 160/4 | **77.0%** | **+36.4%** | Non-Terminal |
| **6 runs + 1 wicket** | 6 | 1 | 148/5 | **17.8%** | **-22.8%** | Non-Terminal |
| **10 runs + 1 wicket** | 10 | 1 | 152/5 | **31.7%** | **-8.9%** | Non-Terminal |
| **15 runs + 1 wicket** | 15 | 1 | 157/5 | **54.4%** | **+13.8%** | Non-Terminal |

### 12.8 CLI Usage Instructions

Run simulation on custom match state:
```bash
python src/simulation/next_over_simulator.py --target_score 190 --current_score 142 --wickets_lost 4 --balls_remaining 24 --overs_completed 16.0
```

Run simulation from an existing match in `data/processed/match_state.csv`:
```bash
python src/simulation/next_over_simulator.py --match_id 430885 --row_index 90
```

Run automated validation test suite (TEST A through TEST E):
```bash
python src/simulation/next_over_simulator.py --run_tests
```

Outputs generated:
- [outputs/next_over_scenarios.csv](file:///c:/Users/HARIHARAN/Desktop/cric/outputs/next_over_scenarios.csv): Tabular records of simulated states, win probabilities, and probability shifts.
- [outputs/next_over_scenario_chart.png](file:///c:/Users/HARIHARAN/Desktop/cric/outputs/next_over_scenario_chart.png): Visual horizontal bar chart illustrating probability shifts against current baseline.

---

## 13. Phase 7: Live Cricket Integration

Phase 7 connects the Cricket Win Predictor to real-time match data via the Cricket Data API ([https://cricketdata.org/](https://cricketdata.org/)).

> [!IMPORTANT]
> The current live predictor is T20-focused and accepts only T20/T20I matches. Other formats returned by the cricket API are filtered out because the trained model uses a 120-ball T20 match-state representation.

### 13.1 Architecture
The live pipeline follows an end-to-end decoupled architecture:
```
Cricket Data API (https://api.cricapi.com/v1/)
       ↓
Live Match Fetcher (`src/live/cricket_api.py`)
       ↓
Normalized Match Representation (`NormalizedMatch`)
       ↓
Live Feature Engineering (`src/live/live_features.py`)
       ↓
Pre-Trained Logistic Regression Model (`models/logistic_regression_win_probability.joblib`)
       ↓
Live Win Probability & Probability Swing Tracker (`src/live/live_match.py`)
```

### 13.2 Live Feature Mapping
The incoming live score object from the API is parsed and converted into the exact 8 features required by the frozen model:
- `target_score`: First innings total runs + 1 (or explicit target if revised/DLS).
- `current_score`: Second innings current runs scored.
- `wickets_lost`: Second innings wickets lost (clamped between 0 and 10).
- `runs_remaining`: $\max(0, \text{target\_score} - \text{current\_score})$.
- `balls_remaining`: $\max(0, 120 - \text{legal balls completed})$.
- `overs_completed`: True fractional overs ($\text{legal balls completed} / 6.0$).
- `current_run_rate`: $\text{current\_score} / \text{overs\_completed}$ (or $0.0$ at start).
- `required_run_rate`: $\text{runs\_remaining} / (\text{balls\_remaining} / 6.0)$ (or $0.0$ if runs or balls $\le 0$).

Cricket overs notation (e.g. `18.4`) is parsed into completed overs (18) and balls in over (4), yielding 112 legal deliveries completed and $18.6667$ fractional overs. Wides and no-balls do not inflate legal balls.

### 13.3 Probability Updates and Swings
- **Win Probability Change**: On each live state change, the tracker calculates $\Delta P = P_t - P_{t-1}$.
- **Swing Terminology**: Reported strictly as "Win Probability Change" or "Win Probability Swing". It reflects changes in the match equation, not metaphysical "momentum".
- **Duplicate Suppression**: The state tracker hashes the core cricket state (score, wickets, legal balls, innings, status). If polling returns an unchanged state, duplicate inferences are skipped.

### 13.4 Terminal & Edge-State Handling
- **Target Reached**: When `current_score >= target_score`, win probability is set deterministically to **100.0%** (loss probability **0.0%**).
- **All Out / Balls Exhausted**: When wickets reach 10 or balls reach 0 before target, win probability is set to **0.0%** (loss probability **100.0%**).
- **First Innings**: If the second innings has not commenced, live prediction is unavailable ("Prediction unavailable: second innings has not started").
- **Unsupported Formats / Super Overs**: Test matches, ODIs, or Super Overs are caught and reported with clear explanatory messages.
- **Abandoned / No Result**: Handled cleanly with an unavailable status.

### 13.5 API Quota & Latency Disclosures
- **Provider**: Cricket Data API (CricAPI v1).
- **Free Tier Quota**: 100 hits per day.
- **Latency**: Free-tier live feeds typically exhibit 30 to 120 seconds latency relative to television broadcasts.
- **Analytical Scope**: This is an analytical estimation tool based on historical patterns, not a guaranteed outcome forecasting or betting system.

### 13.6 Setup and Usage Instructions

1. **Obtain API Key**:
   Register at [cricketdata.org](https://cricketdata.org/) and copy your API key.

2. **Configure Environment**:
   Create a `.env` file in the project root (never committed to version control):
   ```bash
   cp .env.example .env
   ```
   Add your key to `.env`:
   ```ini
   CRICKET_API_KEY=your_actual_api_key_here
   LIVE_POLL_INTERVAL_SECONDS=30
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **List Available Matches**:
   ```bash
   python -m src.live.live_match --list
   ```

5. **Track a Live Match**:
   ```bash
   # Track live match with default polling interval
   python -m src.live.live_match --match_id <MATCH_UUID>

   # Fetch single snapshot and exit
   python -m src.live.live_match --match_id <MATCH_UUID> --once
   ```

6. **Run Test Suite**:
   ```bash
   python tests/test_live.py
   ```

### 13.7 Phase 7.1 — Live T20 Validation

- **Live T20 Availability**: At the time of testing, the real Cricket Data API returned 9 ongoing matches via `currentMatches`, all of which were English County Championship multi-day first-class matches (`format: test`, Day 2 Stumps). All T20 matches returned via `matches` were concluded matches (`matchEnded: true`). Consequently, **no currently live standard T20 match was available during verification**.
- **Validation Outcome**: Reported as `NO_LIVE_MATCH` in accordance with Phase 7.1 protocol. No synthetic or fake live data was fabricated.
- **Overs Notation Investigation**:
  - The API returns overs in standard cricket notation (e.g. `18.4`, `12.5`, `20`).
  - The digit after the decimal indicates completed legal balls in the current over ($18.4 = 18 \text{ overs} + 4 \text{ balls} = 112 \text{ legal balls}$).
  - The previously observed `18.7 overs` was an artifact of formatting true fractional overs ($112 / 6.0 = 18.6667 \approx 18.7$) with `:.1f`.
  - The CLI and normalization layer now preserve both notations clearly: `18.4 ov (18.67 fractional)`.
- **API Quota Tracking**: Real requests consumed 25 of 100 daily quota hits (`hitsToday: 25`, `hitsLimit: 100`).
- **Offline / Mock Verification**: All 19 unit and edge-case tests in `tests/test_live.py` pass cleanly without consuming API hits.

### 13.8 Phase 7.2 — T20-Only Live Match Filtering

To prevent format mismatch errors and ensure alignment with the 120-ball trained model:
- **Strict Format Filter (`is_t20_match`)**: Live match discovery and ingestion filters out all non-T20 formats (Test, ODI, First-Class, List A).
- **Match Listing (`python -m src.live.live_match --list`)**: Displays only T20/T20I fixtures. When only other formats are in progress, it cleanly displays:
  ```text
  No live T20/T20I matches are currently available.
  ```
- **Preserved Invariants**:
  - The 8 model features and 120-ball maximum legal balls definition remain unchanged.
  - Logistic Regression pipeline was not retrained and remains frozen.
  - Cricket overs parsing (`18.4 ov -> 112 legal balls -> 18.67 fractional overs`) is strictly preserved.
  - Test suite expanded to 24 tests covering format filtering, mixed API responses, and absence of live T20s.

---

## 14. Phase 8: T20 Live Dashboard

Phase 8 introduces a professional sports analytics web dashboard connecting the frontend user interface to the live Cricket Data API and the frozen Phase 3 Logistic Regression model.

### 14.1 Technology Stack
- **Frontend**: React + Vite + Tailwind CSS + Recharts + Lucide React
- **Backend**: Python + Flask + Flask-CORS
- **Live Data**: Cricket Data API (v1 REST endpoints, T20/T20I only)
- **Machine Learning**: Scikit-learn Logistic Regression pipeline (`StandardScaler` + `LogisticRegression`)
- **Model Artifact**: Existing validated model (`models/logistic_regression_win_probability.joblib`)

### 14.2 Architecture
```
React Frontend (Vite on :5173)
       ↓  (HTTP REST / Polling every 30s)
Flask Backend (`backend/app.py` on :5000)
       ↓
Live Ingestion & Feature Engineering (`src/live/live_features.py`)
       ↓
Frozen Model (`models/logistic_regression_win_probability.joblib`)
       ↓
Win Probability, Swing Tracker & Next-Over Simulator (`src/simulation/next_over_simulator.py`)
```

### 14.3 Backend Endpoints
- `GET /api/health`: Health status and model verification (`{"status": "ok", "model_loaded": true}`).
- `GET /api/matches`: Retrieves active live T20/T20I matches (strict non-T20 filtering).
- `GET /api/matches/<match_id>`: Evaluates real live match state and generates live win probability.
- `POST /api/simulation`: Evaluates 11 controlled what-if next-over scenarios using the Phase 6 simulation engine.
- `GET /api/demo/matches`: Returns curated demo fixtures for development and simulation testing.
- `GET /api/demo/matches/<demo_id>`: Returns full match state, win probabilities, timeline data, and what-if simulation scenarios.

### 14.4 Demo / No-Live-Match State
- **No Live T20**: When the Cricket Data API contains no live T20/T20I matches, the dashboard cleanly displays:
  ```text
  NO LIVE T20 MATCH AVAILABLE
  There is currently no live T20/T20I match available from the cricket data feed.
  ```
  The dashboard strictly avoids displaying synthetic or fabricated scores as "LIVE".
- **Demo / Simulation Mode**: Users can toggle between the live API feed and the demo simulation mode, which is prominently labeled with a `DEMO / SIMULATION MODE` banner to inspect the full UI, Recharts win probability progression timeline, and what-if scenario simulator.

### 14.5 How to Run the Dashboard

#### 1. Start the Flask Backend
```bash
python backend/app.py
```
*(Runs on `http://127.0.0.1:5000`)*

#### 2. Start the React Frontend
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
*(Runs on `http://localhost:5173` with automated `/api` proxy to the Flask backend)*

#### 3. Run Backend & Live Tests
```bash
python -m unittest discover tests
```
*(Runs all 29 tests covering API client, format filtering, edge cases, and Flask backend endpoints)*

---

## 15. Phase 8.1: Cricket Analytics Command Center UI Redesign

Phase 8.1 refactors the frontend user interface into a high-performance **Cricket Analytics Command Center** combining CrickIQ-style data density, sports analytics typography, and a technical dark aesthetic.

### 15.1 Core Design & Visual Identity
- **Color Palette**: Dark charcoal background (`#090d16` / `#0b0f19`), deep slate surfaces (`slate-950` / `slate-900`), and crisp borders (`slate-800`).
- **Cricket Accents**: Emerald green (`#10b981`) for chasing team, positive probability swings, and healthy rates; Sky cyan (`sky-400`) for technical metrics and legal balls; Rose red (`rose-500`) for defending team and wickets; Amber (`amber-400`) for target and demo modes.
- **Typography**: Clean sans-serif headings with high-contrast tabular monospace figures (`font-mono`) for scores, win percentages, run rates, and ball counts.

### 15.2 Structural Components
1. **Left Sidebar (`frontend/src/components/Sidebar.jsx`)**:
   - Application Brand: `CRICKET WIN PREDICTOR — ANALYTICS COMMAND CENTER`.
   - Anchor Navigation: `OVERVIEW`, `LIVE MATCHES`, `MATCH ANALYSIS`, `PROBABILITY SWINGS`, `WHAT-IF SIMULATOR`.
   - System Telemetry Panel: Model: `Logistic Regression`, `TEST ROC-AUC: 0.9109`, `120-Ball T20 Architecture`, `Cricket Data API`, `Flask :5000`.
   - Responsive overlay drawer on mobile/tablet screens.
2. **Top Header (`frontend/src/components/Header.jsx`)**:
   - Compact command bar with mobile drawer toggle, active match status, live/demo mode switch, and 30s auto-polling indicator with manual refresh.
3. **Match Header (`frontend/src/components/MatchHeader.jsx`)**:
   - Refined sports scoreboard with updated terminology: `CHASING` (2nd innings) vs `DEFENDING`, target equation hub, and runs/balls remaining.
4. **Win Probability Hero (`frontend/src/components/WinProbabilityCard.jsx`)**:
   - Visual centerpiece with prominent percentage typography, dual progress bar, and exact `LATEST SWING [↑/↓ X.X%]` badge.
5. **Match State Card (`frontend/src/components/MatchStateCard.jsx`)**:
   - 6 primary summary cards (`CURRENT SCORE`, `TARGET`, `BALLS REMAINING`, `CURRENT RR`, `REQUIRED RR`, `OVERS`).
   - Expandable drawer: `8 MODEL FEATURES USED` showing the raw input vector ingested by the scikit-learn pipeline.
6. **Win Probability Timeline (`frontend/src/components/ProbabilityTimeline.jsx`)**:
   - Recharts progression curve over completed overs (0 to 20) with 50% par reference line and custom dark tooltip.
7. **Key Probability Swings (`frontend/src/components/ProbabilitySwings.jsx`)**:
   - Delivery-by-delivery swing events list with `EVENT`, `OVER/BALL`, and `DELTA` (`↑`/`↓`), accompanied by an analytical disclaimer.
8. **What-If Next-Over Simulator (`frontend/src/components/NextOverSimulator.jsx`)**:
   - 11 controlled next-over scenario cards showing projected win % and delta from current state, with decision-support disclaimer.
9. **No Live Match State (`frontend/src/components/NoLiveMatchState.jsx`)**:
   - Clean status monitoring display explaining T20-only filtering when no live T20 is active in the API feed.

### 15.3 Strict Scope Preservation
- Zero modifications to ML model weights, scikit-learn pipeline, or feature ordering.
- Zero modifications to Flask API endpoints or backend data contracts.
- Zero synthetic live data; honest separation between `LIVE API` and `DEMO MODE`.
- All 29 backend and live tests pass cleanly.

---

## 16. Phase 8.2: Cricket Analytics Dashboard Light Theme Implementation

Phase 8.2 implements the definitive professional **Light Cricket Analytics Dashboard** visual reference, styled after modern sports broadcast interfaces.

### 16.1 Design Language & Palette
- **Surfaces**: Crisp white cards (`#FFFFFF`) on light blue-grey canvas (`#F5F8FB`) with subtle borders (`#E3EAF0`).
- **Cricket Green**: `#0B9F72` (primary) and `#EAF8F2` (light background) for chasing team, positive probability swings, and healthy indicators.
- **Defending Red / Coral**: `#EF5B67` and `#FFF0F2` (soft negative background) for defending team, wickets, and negative deltas.
- **Typography**: Professional sans-serif headings with high-contrast monospace tabular figures (`font-mono`) for scores, balls, overs, run rates, and win probabilities.

### 16.2 Key Interface Elements
1. **Horizontal Header (`Header.jsx`)**:
   - Clean white top bar with cricket ball emblem, `CRICKET WIN PREDICTOR · T20 ANALYTICS`, horizontal navigation links (`Live Matches`, `Match Analysis`, `Swings`, `Simulator`), `LIVE / DEMO` toggle, `Poll 30s` badge, refresh and settings buttons.
2. **Horizontal Live Match Strip (`MatchSelector.jsx`)**:
   - Prominent match strip directly below the header with quick cards (e.g. `IND vs AUS · T20 · Live · 16.0 ov`) highlighting the active match in cricket green.
3. **Match Hero / Scoreboard (`MatchHeader.jsx`)**:
   - Large horizontal scoreboard with subtle stadium gradient wash, circular team crests (`IND`, `AUS`), `CHASING` / `DEFENDING` pills, and central target equation hub (`TARGET 190 · NEED 48 RUNS FROM 24 BALLS · CRR 8.88 · RRR 12.00`).
4. **Win Probability Hero (`WinProbabilityCard.jsx`)**:
   - Dominant percentage numbers (`India 40.6%` vs `Australia 59.4%`), dual green/red probability bar, and `LATEST SWING ↓ 4.2%` pill with context.
5. **Match State Card (`MatchStateCard.jsx`)**:
   - Unified single horizontal card with 6 metric columns (`CURRENT SCORE`, `TARGET`, `RUNS NEEDED`, `BALLS REMAINING`, `CURRENT RR`, `REQUIRED RR`) and an expandable `8 MODEL FEATURES USED` drawer.
6. **Win Probability Over Time (`ProbabilityTimeline.jsx`)**:
   - Dual-line Recharts visualization (India green, Australia red) with 50% par dashed reference line and light hover tooltip.
7. **Key Probability Swings (`ProbabilitySwings.jsx`)**:
   - Delivery-by-delivery leverage events (`FOUR`, `WICKET`, `SIX`, `DOT BALL`) with event badges, delta chips (`+4.8%`, `-7.4%`), and model disclaimer.
8. **What-If Next-Over Simulator (`NextOverSimulator.jsx`)**:
   - 11 scenario tiles with projected win % and change from current state, interactive scenario selection, and "View all 11 scenarios" toggle.
9. **Model Details Card (`ModelDetailsCard.jsx`)**:
   - Clean specifications card displaying `Logistic Regression`, `TEST ROC-AUC: 0.9109`, `120-Ball T20 Architecture`, etc.

### 16.3 Strict Logic Preservation
- 100% untouched ML model weights, feature pipeline, and inference logic.
- 100% untouched Flask backend endpoints.
- All 29 backend and live-client tests pass.

---

## 17. Phase 8.2.1: Reference Fidelity Polish

Phase 8.2.1 performs a precision visual polish pass on the Phase 8.2 light-theme dashboard to achieve complete alignment with the broadcast analytics reference design.

### 17.1 Key Visual Refinements
- **Header**: `Live Matches` is configured as the default active navigation tab with subtle `#EAF8F2` background, cricket green `#0B9F72` text, and rounded active indicator. Control pills have been balanced with reduced visual weight.
- **Live Match Strip**: Match cards now display the chase situation explicitly (e.g., `AUSTRALIA vs INDIA · T20 · LIVE · 16.0 OV · India need 48 from 24`) with green live status dots.
- **De-emphasized Demo Mode Banner**: The saturated warning-style banner has been replaced with a pale warm card (`#F7F5F0`, border `#E8E4DC`) and clean `DEMO MATCH` badge, remaining clearly informative without visual alarm.
- **Match Hero / Scoreboard**: Enhanced with a subtle cricket stadium pitch arc silhouette texture, larger 56px team emblems, and a distinct `#EEF5FC` target centerpiece anchoring the chase equation.
- **Win Probability Hero**: Vertical whitespace has been compressed for density, integrating the `LATEST SWING [↑/↓ X.X%]` indicator into the bottom composition.
- **Match State Card**: Enhanced numerical typography in monospace with clear subtle dividers.
- **Probability Timeline**: Styled with an `#A8B4C2` 50% par line and crisp tooltip styling.
- **Key Probability Swings**: Formatted in cricket broadcast style with delivery-level event badges (`FOUR`, `WICKET`, `SIX`, `DOT BALL`) and directional arrow deltas (`+4.8% ↑`, `-7.4% ↓`).
- **Next-Over Simulator**: Refined scenario tiles and selected outcome banner with clear cricket decision-support styling.

---

## 18. Phase 9: Live T20 Match Validation & Data Reliability

Phase 9 validates and hardens the real-time live-data pipeline from API discovery down to model inference, probability swing calculation, duplicate protection, and dashboard synchronization.

### 18.1 Pipeline Flow
```
Cricket Data API (v1 /currentMatches)
        ↓
Strict T20/T20I Format Filtering
        ↓
Second-Innings State Extraction (120-ball chase)
        ↓
Exact 8 Model Feature Vector
        ↓
Frozen Logistic Regression Model (models/logistic_regression_win_probability.joblib)
        ↓
Live Win Probability & Dual Complements
        ↓
Consecutive Delta & Swing Calculation (percentage points)
        ↓
Duplicate State Hash Suppression
        ↓
Dashboard Real-Time Polling & Refresh
```

### 18.2 Reliability Specifications
- **Live API Source**: Cricket Data API (`https://api.cricapi.com/v1/`).
- **T20/T20I Filtering**: Strict case-insensitive and whitespace-safe matching (`t20`, `T20`, `t20i`, `T20I`). Tests, ODIs, First-Class, List A, and Super Overs are strictly rejected.
- **Polling Interval**: Standard 30-second interval via backend gateway to protect API rate limits and avoid duplicate calls.
- **State Mapping & Eight Features**:
  1. `target_score` (1st innings score + 1)
  2. `current_score` (cumulative chasing runs)
  3. `wickets_lost` (0 to 10)
  4. `runs_remaining` (target - current)
  5. `balls_remaining` (120 - legal deliveries bowled)
  6. `overs_completed` (true fractional overs: `comp_overs + balls_in_over / 6.0`)
  7. `current_run_rate` (`current_score / overs_completed`)
  8. `required_run_rate` (`runs_remaining / (balls_remaining / 6.0)`)
- **Duplicate State Suppression**: In-memory state signatures track match ID, innings, score, wickets, and legal deliveries bowled. Identical scoreboards polled consecutively produce `is_new = False` with 0.0 delta, eliminating artificial swing events.
- **Terminal Handling**:
  - Target reached $\rightarrow$ Chasing win probability = 100% (1.0).
  - All out before target $\rightarrow$ Chasing win probability = 0% (0.0).
  - Balls exhausted before target $\rightarrow$ Chasing win probability = 0% (0.0).
- **Stale Data & Quota Protection**: Single server-side API client; credentials are never leaked to frontend.
- **API Error Handling**: Network timeouts, HTTP 429 rate limits, and malformed JSON return structured status responses without dashboard crashing.

### 18.3 Validation Result
- **Status**: `VERIFIED_STANDBY` (`NO_LIVE_T20_AVAILABLE`).
- **Validation Log**: `outputs/live_t20_validation_log.csv`
- **Validation Report**: `outputs/live_t20_validation_report.json`
- **Note**: *"No live T20/T20I match was available during validation; no synthetic live data was used."* During API discovery, all 9 current matches returned were County Championship first-class Test matches, which were correctly filtered out by the T20 validator. All 8 reliability criteria were deterministically validated with the frozen model across 49 automated unit tests.

---

## 19. Phase 9.1: Live T20 Data Source Debugging & Coverage Audit

Phase 9.1 performs an exhaustive diagnostic audit of the live data provider (Cricket Data API / CricAPI v1) to investigate why an active real-world T20 fixture—**Nigeria vs Sierra Leone** (Match 12, Quadrangular T20I Series in Nigeria 2026, 27 September 2026)—was not discovered by the dashboard.

### 19.1 Audit Methodology & Preserved Raw Payloads
Every queried endpoint was audited before project-side normalization or filtering, with all API credentials sanitized:
- **`GET /currentMatches?offset=0`**: 10 matches returned. 9 County Championship matches (`format: test`) and 1 ODI (`India vs West Indies, 1st ODI`). Total rows available = 10. Zero T20 fixtures present.
- **`GET /matches?offset=0`**: 25 matches returned. Primarily historical matches from August–September 2026 (e.g. England vs Pakistan Tests, Luxembourg vs Belgium Women T20Is, Delhi Premier League). Zero Nigeria or Sierra Leone fixtures.
- **`GET /matches?offset=25`**: 25 matches returned. Historical matches extending backwards into July 2026 (Namibia Quadrangular, Switzerland Quadrangular, Sheffield Shield). Zero Nigeria or Sierra Leone fixtures.
- **`GET /series?search=Nigeria`**: Located 4 registered series, including `Quadrangular T20I Series in Nigeria, 2026` (`series_id: c1447761-1361-426a-8735-d7adaa73f407`).
- **`GET /series_info?id=c1447761-1361-426a-8735-d7adaa73f407`**: Deep-inspected tournament metadata and match schedule.

### 19.2 The Six Distinction Criteria
The audit programmatically evaluated all six diagnostic hypotheses:
1. **Case 1: Match absent from raw API response**: **CONFIRMED (TRUE)**. Neither Nigeria nor Sierra Leone appears in any match record across `/currentMatches` (offset 0), `/matches` (offset 0), or `/matches` (offset 25).
2. **Case 2: Match present but rejected by project filtering**: **FALSE**. The fixture was never returned by the provider and therefore was not discarded by `is_t20_match` or format checking.
3. **Case 3: Match present but status interpreted incorrectly**: **FALSE**. No raw entry exists whose status, innings, or `matchStarted` flag was misinterpreted.
4. **Case 4: Match present only at another pagination offset**: **FALSE**. `/currentMatches` contains only 10 total matches (`totalRows: 10`). `/matches` pagination moves further back in time (July 2026) as offset increases.
5. **Case 5: Match present in `/matches` but absent from `/currentMatches`**: **FALSE**. The match is absent from the general `/matches` catalogue as well.
6. **Case 6: Match present in `/match_info` but not discoverable through listing endpoints**: **FALSE**. Direct query of the tournament schedule (`/series_info?id=c1447761-1361-426a-8735-d7adaa73f407`) returned `"matches": 0`, `"squads": 0`, and `"matchList": []`. The provider has created a series container but has not indexed or scheduled any match IDs.

### 19.3 Audit Conclusion: `LIVE_PROVIDER_COVERAGE_LIMITATION`
- **Result**: `LIVE_PROVIDER_COVERAGE_LIMITATION`
- **Audit Artifact**: `outputs/live_api_coverage_audit.json`
- **Root Cause**: While external consumer scoreboards report Nigeria vs Sierra Leone as live, the commercial data provider (CricAPI v1) does not index or provide ball-by-ball scorecards for the Nigeria Quadrangular T20I tournament.
- **Architectural Integrity Maintained**:
  - No synthetic live data was fabricated.
  - No external web scraping was introduced.
  - The T20 filter was NOT relaxed to accept Test or ODI fixtures.
  - All 49 backend and live tests pass (100%).
  - Frontend production build compiles with zero errors.

---

## 20. Phase 10: Production Deployment & Operational Readiness

Phase 10 hardens the Cricket Win Predictor application for production deployment, auditing environment security, CORS configuration, API quota safeguards, live state reliability, and deployment procedures.

### 20.1 End-to-End Production Architecture
```
[External Live Data Provider]
Cricket Data API (https://api.cricapi.com/v1)
              │
              ▼ (HTTPS / API Key secured server-side)
[Flask REST Backend] (Port 5000 / Render / Railway)
├── CORS Origin Guard (`FRONTEND_ORIGIN`)
├── Live Match Discovery & Ingestion
├── Strict T20/T20I Format Filtering (`is_t20_match`)
├── 120-Ball Second-Innings State Feature Mapping
├── Frozen ML Pipeline (`models/logistic_regression_win_probability.joblib`)
├── Probability Swing Tracker & Duplicate Suppression Signature
└── Next-Over What-If Simulator (`src/simulation/next_over_simulator.py`)
              │
              ▼ (JSON REST / Public CORS / Polling 30s)
[React 19 + Vite Dashboard] (Vercel / Netlify / Static Hosting)
├── Light Sports Broadcast UI (Phase 8.2.1 LOCKED)
├── Dynamic API Base URL (`VITE_API_BASE_URL`)
├── Horizontal Scoreboard Hero & Win Probability Bar
├── Dual-Line Recharts Probability Timeline & Leverage Swings
└── Isolated Demo / Simulation Mode
```

### 20.2 Security & Credential Isolation Audit
- **Strict Server-Side Key Isolation**: `CRICKET_API_KEY` is loaded exclusively inside the Flask backend process. The React client has zero access to and never receives the provider API key.
- **Credential Masking**: All outgoing URLs, network logs, and error traces systematically mask API keys via `mask_sensitive_url()` (`apikey=***REDACTED***`).
- **Repository Cleanliness**: Verified repo-wide search returned **0 raw API key occurrences** in tracked files.
- **Git Ignore Protection**: `.gitignore` strictly blocks `.env`, `.env.*`, with an explicit whitelist exception for `.env.example`.

### 20.3 Deployment Configuration & Environment Variables

#### Frontend (e.g. Vercel, Netlify)
| Variable | Value | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `https://your-backend.onrender.com` | Base URL of deployed Flask backend (omitted in local development to use Vite `/api` proxy). |

- **Build Command**: `cd frontend && npm install && npm run build`
- **Output Directory**: `frontend/dist`

#### Backend (e.g. Render, Railway, Linux Container)
| Variable | Value | Description |
|---|---|---|
| `CRICKET_API_KEY` | `<your-cricapi-key>` | Commercial Cricket Data API key (kept confidential). |
| `LIVE_POLL_INTERVAL_SECONDS` | `30` | Cadence for live match polling. |
| `FRONTEND_ORIGIN` | `https://your-frontend.vercel.app` | Production origin allowed through CORS. Set to `*` for initial dev/preview. |
| `PORT` | `5000` | Server listening port (assigned dynamically by platform). |
| `FLASK_DEBUG` | `false` | Disables debug mode in production. |

- **Startup Command (Production WSGI)**:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:$PORT backend.app:app
  ```
- **Startup Command (Direct Python / Local)**:
  ```bash
  python backend/app.py
  ```

### 20.4 Live Provider Reliability & Quota Protection
- **Single Gateway**: Only the backend contacts the Cricket Data API. Browser clients poll the backend, never the external provider directly.
- **Quota Safeguard**: 30-second interval ensures single-match tracking uses at most ~2 hits/minute during active play, honoring the 100 hits/day free tier.
- **Stale State Handling**: Match details return `is_cached: true/false` and an ISO UTC timestamp `last_updated`. If the provider feed disconnects, the frontend displays an informative warning banner without fabricating scores.
- **Live Coverage Caveat**: As audited in Phase 9.1 (`LIVE_PROVIDER_COVERAGE_LIMITATION`), provider coverage varies across regional/associate tournaments. If a match is absent from CricAPI, the dashboard honestly displays the no-live-match standby state and allows seamless exploration in Demo Mode.

### 20.5 Verification Results
- **Automated Tests**: **56/56 passing** (`python -m unittest discover tests`).
- **Production Build**: **0 errors** (`npm run build`).
- **Local Smoke Tests**: **6/6 endpoints verified** (`outputs/smoke_test.py`).
- **Audit Report**: [`outputs/phase10_production_readiness_report.json`](outputs/phase10_production_readiness_report.json).

