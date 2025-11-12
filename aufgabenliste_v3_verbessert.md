# Projekt SuRe-Ferment: Gesamt-Aufgabenliste (Version 3, Vollständig Korrigiert)

**Korrektur-Log:**
- Version 1: Original
- Version 2: Kritische Logikfehler behoben (Scaler-Loop, Budget-Tracking, Ensemble-Tests)
- **Version 3: Ausführbarkeits-Verbesserungen** (CLI-Parameter, Logging-Setup, Budget-Strategie, Explizite Pfade)

---

## 1. Paket: Setup & Qualitätssicherung ✅ ERLEDIGT

### 1.1 Projektstruktur & Umgebungs-Setup
* `[x]` **Aufgabe 1.1.1 (Struktur):** Projektverzeichnis `SURROGATE-FERMENT` anlegen.
* `[x]` **Aufgabe 1.1.2 (Sub-Ordner):** Unterordner erstellen: `src/`, `data/`, `models/`, `notebooks/`, `tests/`.
* `[x]` **Aufgabe 1.1.3 (Umgebung):** `environment.yml` (Conda) oder `requirements.txt` (pip) erstellen.
* `[x]` **Aufgabe 1.1.4 (Basis-Pakete definieren):** `environment.yml` füllen (python=3.10, numpy, scipy, torch, gymnasium, stable-baselines3, pytest, pandas, wandb/tensorboard).
* `[x]` **Aufgabe 1.1.5 (Versionskontrolle):** `git init` und `.gitignore`-Datei anlegen.

### 1.2 Konfigurations-Management
* `[x]` **Aufgabe 1.2.1 (Config-Datei):** Zentrale `src/config.py` oder `config.yaml` erstellen.
* `[x]` **Aufgabe 1.2.2 (Pfade definieren):** Alle Pfade (data/, models/) in die Config aufnehmen.
* `[x]` **Aufgabe 1.2.3 (HF-Modell Parameter):** Parameter (MU_MAX, K_S, YXS, DT) in Config festlegen.
* `[x]` **Aufgabe 1.2.4 (Budget definieren):** `HF_BUDGET_SECONDS = 8 * 3600` und `INITIAL_DATASET_EPISODES = 100` festlegen.
* `[x]` **Aufgabe 1.2.5 (Modell-Parameter):** Hyperparameter (SURROGATE_HIDDEN_LAYERS, LEARNING_RATE, PPO_STEPS, GAMMA) in Config definieren.
* `[x]` **Aufgabe 1.2.6 (Reward-Gewichte):** Multi-Objektiv-Gewichte in Config definieren (z.B. `W_BIOMASSE = 1.0`, `W_SUBSTRAT_KOSTEN = 0.5`).

### 1.3 Daten-Management & Serialisierung
* `[x]` **Aufgabe 1.3.1 (Datenformat):** Speicherformat festlegen (z.B. pandas.DataFrame als `.pkl` oder `.parquet`).
* `[x]` **Aufgabe 1.3.2 (Namenskonvention):** Benennung definieren (z.B. `D_v1_initial.pkl`, `scaler_state_v1.pkl`, `policy_v1.zip`).
* `[x]` **Aufgabe 1.3.3 (Helper-Funktionen):** `save_dataset`, `load_dataset`, `save_model`, `load_model`, `save_scaler`, `load_scaler` in `src/utils.py` implementieren.
* `[x]` **Aufgabe 1.3.4 (Budget Tracker):** Implementiere eine `BudgetTracker` Klasse in `src/utils.py`.
    * `[x]` Klasse liest/schreibt den aktuellen Stand (verbrauchte_zeit, verbrauchte_queries) aus/in eine `data/budget_state.json`.
    * `[x]` Bietet Methoden wie `check_budget(required_time)` (gibt True/False zurück), `update_budget(spent_time)`, `reset()`.

### 1.4 Initiales Test-Setup (Qualitätssicherung)
* `[x]` **Aufgabe 1.4.1 (Pytest-Struktur):** `tests/` initialisieren (z.B. `tests/test_config.py`).
* `[x]` **Aufgabe 1.4.2 (Test: Config-Laden):** Test schreiben, der prüft, ob Config-Datei geladen werden kann.
* `[x]` **Aufgabe 1.4.3 (Test: Daten-I/O):** Test in `tests/test_utils.py` schreiben (Dummy-Array/Scaler speichern/laden/vergleichen).
* `[x]` **Aufgabe 1.4.4 (Test: Budget Tracker):** Test in `tests/test_utils.py` schreiben, der `update_budget` aufruft, speichert, neu lädt und den Zählerstand prüft.

### 1.5 Logging & Experiment-Tracking Setup
* `[x]` **Aufgabe 1.5.1 (Logging-Tool wählen):** Entscheide zwischen WandB oder TensorBoard (aus 1.1.4) basierend auf Präferenz.
* `[x]` **Aufgabe 1.5.2 (Logger-Utility):** Implementiere `src/logger.py` mit einer `ExperimentLogger`-Klasse.
    * `[x]` Initialisiert Logging (z.B. `wandb.init(project="surrogate-ferment")` oder `SummaryWriter(log_dir="runs/")`).
    * `[x]` Bietet Methoden: `log_metric(name, value, step)`, `log_config(config_dict)`, `log_model(model_path)`.
* `[x]` **Aufgabe 1.5.3 (Integration):** Alle Trainings-Skripte (S1.5, R1.4, 5.1) sollen `ExperimentLogger` nutzen, um Metriken und Konfigurationen zu protokollieren.

---

## 2. Paket: Phase 1 (Fundament & Datengrundlage) ✅ ERLEDIGT

### 2.1 [WP H0] High-Fidelity Data Source
* `[x]` **Aufgabe H0.1 (Recherche):** 2-3 etablierte ODE-Systeme für Fermentation recherchieren (da kein Modell vorhanden).
* `[x]` **Aufgabe H0.2 (Implementierung Dynamik):** `src/hf_model.py` erstellen. `hf_dynamics(t, y, action, params)` implementieren (nutzbar von `scipy.integrate.solve_ivp`).
* `[x]` **Aufgabe H0.3 (Implementierung Episoden-Simulator):** `simulate_episode(initial_state, action_policy_func, horizon, params)` in `src/hf_model.py` implementieren.
    * `[x]` Diese Funktion prüft KEIN Budget (das macht der Aufrufer).
    * `[x]` Sie gibt die `trajectory` UND die `elapsed_time` (Wand-Zeit in Sekunden) zurück.
* `[x]` **Aufgabe H0.4 (Stabilitäts-Tests):** `tests/test_hf_model.py` erstellen (Test mit Null-Aktionen, Test mit Max-Aktionen, Prüfung auf Plausibilität/NaNs).
* `[x]` **Aufgabe H0.5 (Budget-Benchmark):** `src/benchmark_hf.py` erstellen. Misst Wand-Zeit für 10 Episoden, rechnet hoch auf 8h-Budget und speichert `MAX_HF_BUDGET_EPISODES` in Config.
* `[x]` **Aufgabe H0.6 (Initial-Datenset-Policy):** `src/data_builder.py` erstellen. Explorative Policy definieren (z.B. `RandomActionPolicy`).
* `[x]` **Aufgabe H0.7 (Initiales Datenset generieren):** `main`-Funktion in `src/data_builder.py` implementieren.
    * `[x]` Skript akzeptiert CLI-Argumente: `--output_path`, `--n_episodes`, `--config_path`.
    * `[x]` Instanziiert `BudgetTracker` (1.3.4).
    * `[x]` VOR jeder Episode: `budget_tracker.check_budget(estimated_time)`.
    * `[x]` Ruft `simulate_episode` (H0.3) N-mal auf, erhält `(trajectory, elapsed_time)`.
    * `[x]` NACH jeder Episode: `budget_tracker.update_budget(elapsed_time)`.
    * `[x]` Sammelt alle `(state, action, next_state, reward)` Tupel.
* `[x]` **Aufgabe H0.8 (Datenset-Verarbeitung & Speicherung):** Gesammelte Tupel (H0.7) in DataFrame formatieren und als `data/D_v1_initial.pkl` (1.3.2) speichern.

### 2.2 [WP S1] Surrogate Model (Baseline)
* `[x]` **Aufgabe S1.1 (Daten-Scaler fitten):**
    * `[x]` Erstelle `src/fit_scaler.py` als ausführbares CLI-Skript.
    * `[x]` Skript akzeptiert Argumente: `--dataset_path`, `--output_dir`, `--version`.
    * `[x]` Lade das Datenset (z.B. `D_v1_initial.pkl`).
    * `[x]` Implementiere/Nutze `Scaler`-Klassen (z.B. `StandardScaler` für States, `MinMaxScaler` für Aktionen).
    * `[x]` Fitte `state_scaler` auf `state` und `next_state` Spalten (kombiniert).
    * `[x]` Fitte `action_scaler` auf `action` Spalten.
    * `[x]` Speichere beide Scaler (1.3.3) als `models/scaler_state_v{version}.pkl` und `models/scaler_action_v{version}.pkl`.
* `[x]` **Aufgabe S1.2 (PyTorch Dataset-Klasse):** `src/surrogate_model.py` erstellen. `FermentationDataset(torch.utils.data.Dataset)` implementieren.
    * `[x]` `__init__` lädt Datenset und *beide* Scaler (S1.1).
    * `[x]` `__getitem__` wendet `state_scaler` auf `state`/`next_state` und `action_scaler` auf `action` an (EXPLIZIT definiert, nicht "eventuell").
* `[x]` **Aufgabe S1.3 (Surrogat-Architektur):** `SurrogateModel(nn.Module)` in `src/surrogate_model.py` implementieren (z.B. MLP, Architektur aus Config 1.2.5).
* `[x]` **Aufgabe S1.4 (Trainings-Skript-Setup):** `src/train_surrogate.py` erstellen.
    * `[x]` Skript akzeptiert CLI-Argumente: `--dataset_path`, `--state_scaler_path`, `--action_scaler_path`, `--output_dir`, `--version`, `--ensemble_index` (optional, für S2.1).
    * `[x]` Lädt Config/Daten/Scaler, instanziiert Dataset/Loader/Modell/Optimizer/Loss.
* `[x]` **Aufgabe S1.5 (Trainings-Loop):** Trainings-Loop in `src/train_surrogate.py` implementieren (inkl. Logging mit `ExperimentLogger` (1.5.2) und Best-Model-Saving).
    * `[x]` Wenn `--ensemble_index` gegeben: Führe Bootstrap-Sampling (Ziehen mit Zurücklegen) auf dem Trainings-Datenset durch.
* `[x]` **Aufgabe S1.6 (Modell speichern):** Bestes Modell (S1.5) speichern.
    * `[x]` Wenn `--ensemble_index` gegeben: `models/surrogate_v{version}_ens_{index}.pth`.
    * `[x]` Sonst: `models/surrogate_v{version}_best.pth`.
* `[x]` **Aufgabe S1.7 (One-Step-Validierungstest):** `tests/test_surrogate.py` erstellen (lädt Modell/Scaler, macht 1 Vorhersage, ent-skaliert, prüft MSE-Fehler < Schwellenwert).
* `[x]` **Aufgabe S1.8 (Test: Ensemble-Diversität):** Test in `tests/test_surrogate.py` hinzufügen.
    * `[x]` Trainiere ein Mini-Ensemble (M=2) auf Bootstrap-Samples.
    * `[x]` `assert`e, dass die Varianz ihrer Vorhersagen für denselben Input > 1e-6 ist.

---

## 3. Paket: Phase 2 (RL-Umgebung & Policy-Training) ✅ ERLEDIGT

### 3.1 [WP S3] Surrogate-Env Wrapper
* `[x]` **Aufgabe S3.1 (Datei anlegen):** `src/surrogate_env.py` erstellen.
* `[x]` **Aufgabe S3.2 (Klassen-Definition):** `SurrogateEnv(gym.Env)` implementieren.
* `[x]` **Aufgabe S3.3 (Implementierung `__init__`):**
    * `[x]` `__init__(self, surrogate_model_path, state_scaler_path, action_scaler_path, initial_states=None)` definieren.
    * `[x]` Lade Config (1.2.1) für Dimensionen/Horizon.
    * `[x]` Lade das Surrogat-Modell (aus `surrogate_model_path`) und setze in `eval()`-Modus.
    * `[x]` Lade den `state_scaler` (aus `state_scaler_path`).
    * `[x]` Lade den `action_scaler` (aus `action_scaler_path`).
    * `[x]` Wenn `initial_states` gegeben: nutze diese. Sonst: Lade realistische Initialzustände aus einem Datenset (Pfad aus Config).
    * `[x]` Definiere `action_space` / `observation_space`.
* `[x]` **Aufgabe S3.4 (Implementierung `reset`):** Wählt zufälligen Startzustand, setzt `self.current_step = 0`, gibt `(self.state, {})` zurück.
* `[x]` **Aufgabe S3.5 (Implementierung `_calculate_reward`):** `_calculate_reward(self, state, action, next_state)` implementieren.
    * `[x]` Multi-Objektiv: `Reward = (w1 * biomasse_zuwachs) - (w2 * substrat_kosten)`.
    * `[x]` Gewichte `w1/w2` aus Config (1.2.6) laden.
* `[x]` **Aufgabe S3.6 (Implementierung `step` - KERN):**
    1.  Skaliere `self.state` (mit `state_scaler`) und `action` (mit `action_scaler`).
    2.  Inferiere skalierten `predicted_next_state` mit Surrogat-Modell (mit `torch.no_grad()`).
    3.  Ent-skaliere `predicted_next_state` (mit `state_scaler.inverse_transform`).
    4.  Setze `self.state = unscaled_predicted_next_state`.
    5.  Berechne Reward (S3.5) mit un-skalierten Werten.
    6.  Prüfe `terminated = self.current_step >= self.horizon`. `truncated = False`.
    7.  Return `(self.state, reward, terminated, truncated, {})`.
* `[x]` **Aufgabe S3.7 (Umgebungs-Check):** `tests/test_surrogate_env.py` erstellen. `check_env(SurrogateEnv(model_path, state_scaler_path, action_scaler_path))` aufrufen (mit Pfaden zu `v1`-Artefakten).

### 3.2 [WP R1] RL Training on Surrogate
* `[x]` **Aufgabe R1.1 (Datei anlegen):** `src/train_rl.py` erstellen.
* `[x]` **Aufgabe R1.2 (Setup & Initialisierung):**
    * `[x]` Skript akzeptiert CLI-Argumente: `--surrogate_path`, `--state_scaler_path`, `--action_scaler_path`, `--save_path`, `--config_path`.
    * `[x]` Instanziiere `env = SurrogateEnv(surrogate_path, state_scaler_path, action_scaler_path)`.
    * `[x]` Wrappe `env = Monitor(env)`.
* `[x]` **Aufgabe R1.3 (Modell-Definition):** `PPO("MlpPolicy", env, ...)` instanziieren. Alle Hyperparameter aus Config (1.2.5) laden.
* `[x]` **Aufgabe R1.4 (Training):** `model.learn(total_timesteps=TOTAL_TIMESTEPS)` aufrufen (Timesteps aus Config). Nutze `ExperimentLogger` (1.5.2) für Logging.
* `[x]` **Aufgabe R1.5 (Modell speichern):** `model.save(save_path)`.
* `[x]` **Aufgabe R1.6 (Smoke-Test):** `tests/test_rl_training.py` erstellen. Modell (R1.5) laden, `model.predict(obs)` ausführen.

---

## 4. Paket: Phase 3 (Validierung & Active Learning Loop)

### 4.1 [WP V1] Validation on High-Fidelity
* `[ ]` **Aufgabe V1.1 (Skript erstellen):** `src/evaluate_policy.py` erstellen.
    * `[ ]` Skript akzeptiert CLI-Argumente: `--policy_path`, `--policy_type` (rl/baseline), `--n_episodes`, `--use_budget` (True/False).
* `[ ]` **Aufgabe V1.2 (Setup):** RL-Policy (R1.5) laden (falls `policy_type=rl`), `hf_dynamics` (H0.2) laden, Reward-Logik (S3.5) importieren.
    * `[ ]` Wenn `use_budget=True`: `BudgetTracker` (1.3.4) instanziieren.
* `[ ]` **Aufgabe V1.3 (Evaluierungs-Helper):** `run_hf_episode(policy, hf_dynamics, params, initial_state, horizon, budget_tracker=None)` implementieren.
    * `[ ]` VOR der Episode: Wenn `budget_tracker` gegeben: `budget_tracker.check_budget(estimated_time)`.
    * `[ ]` Iteriert `horizon` Schritte:
        - `action = policy.predict(current_state, deterministic=True)` (oder Baseline-Logik).
        - `next_state = solve_ivp(hf_dynamics, ...)` (für 1 Zeitschritt dt).
        - `reward = calculate_reward(current_state, action, next_state)`.
        - `current_state = next_state`.
    * `[ ]` NACH der Episode: Wenn `budget_tracker` gegeben: `budget_tracker.update_budget(actual_time)`.
    * `[ ]` Gibt `total_reward` zurück.
* `[ ]` **Aufgabe V1.4 (Haupt-Evaluierungs-Loop):** `N_EVAL_EPISODES` (aus Config oder CLI) ausführen, `run_hf_episode` (V1.3) aufrufen, Rewards sammeln.
* `[ ]` **Aufgabe V1.5 (Reporting):** `mean()` und `std()` der Rewards (V1.4) berechnen, ausgeben und in Datei speichern (z.B. `results/eval_policy_v{version}.json`).

### 4.2 [WP S2] Active Learning Loop (Kernforschung)
* `[ ]` **Aufgabe S2.1 (Ensemble-Training):** `src/train_surrogate.py` (S1.4) mehrfach aufrufen, um ein Ensemble von `M=5` Modellen zu trainieren.
    * `[ ]` Für `i in range(M)`: Rufe `python src/train_surrogate.py --dataset_path ... --ensemble_index {i}` auf.
    * `[ ]` Jedes Modell auf einem anderen Bootstrap-Sample (in S1.5 implementiert) trainieren.
    * `[ ]` Alle `M` Modelle speichern (z.B. `..._ens_0.pth`, `..._ens_1.pth`).
* `[ ]` **Aufgabe S2.2 (Unsicherheits-Metrik):** `get_ensemble_uncertainty(state, action, ensemble_models, state_scaler, action_scaler)` in `src/surrogate_model.py` implementieren.
    * `[ ]` Skaliert Input mit *beiden* Scalern.
    * `[ ]` Führt Inferenz auf *allen* `M` Modellen durch.
    * `[ ]` Berechnet Varianz (oder Std) über die `M` ent-skalierten `next_state`-Vorhersagen.
    * `[ ]` Gibt Unsicherheits-Score zurück (z.B. mittlere Varianz über alle State-Dimensionen).
* `[ ]` **Aufgabe S2.3 (Skript erstellen):** `src/active_learning.py` erstellen.
    * `[ ]` Skript akzeptiert CLI-Argumente: `--dataset_path`, `--policy_path`, `--ensemble_dir`, `--state_scaler_path`, `--action_scaler_path`, `--n_queries`, `--output_path`.
* `[ ]` **Aufgabe S2.4 (Kandidaten-Generierung):** `generate_candidate_queries(policy_path, surrogate_path, state_scaler_path, action_scaler_path, n_episodes)` implementieren.
    * `[ ]` Instanziiert `SurrogateEnv(surrogate_path, state_scaler_path, action_scaler_path)`.
    * `[ ]` Lädt Policy und lässt sie für `K` Episoden (z.B. `K=20` aus Config) in der Surrogat-Umgebung laufen.
    * `[ ]` Sammelt *alle* `(state, action)`-Paare, die die Policy während dieser Episoden besucht.
* `[ ]` **Aufgabe S2.5 (Query-Selektion):** `select_best_queries(candidates, ensemble_models, state_scaler, action_scaler, n_queries)` implementieren.
    * `[ ]` Nimmt die Kandidaten (S2.4) und berechnet für jeden Kandidaten den Unsicherheits-Score (mit S2.2).
    * `[ ]` Sortiert die Kandidaten nach dem Unsicherheits-Score (absteigend).
    * `[ ]` Gibt die Top-`n_queries` `(state, action)`-Paare mit der höchsten Unsicherheit zurück.
* `[ ]` **Aufgabe S2.6 (HF-Abfrage & Budget-Prüfung):** `query_hf_model(queries, hf_dynamics, params, budget_tracker)` implementieren.
    * `[ ]` VOR den Abfragen: Schätzt Gesamtzeit (`len(queries) * avg_time_per_single_step`) und ruft `budget_tracker.check_budget(estimated_time)` auf.
    * `[ ]` Falls Budget nicht ausreicht: Reduziere `queries` auf machbare Anzahl oder gib Warnung zurück.
    * `[ ]` Für jeden Query-Punkt `(state, action)`:
        - Ruft `solve_ivp` mit `hf_dynamics` für **einen** Zeitschritt auf, um `next_state` zu erhalten.
        - Berechnet `reward` mit Reward-Logik (S3.5).
    * `[ ]` NACH allen Abfragen: `budget_tracker.update_budget(actual_total_time)` aufrufen.
    * `[ ]` Sammelt neue `(state, action, next_state, reward)`-Tupel und gibt sie zurück.
* `[ ]` **Aufgabe S2.7 (Datenset-Update):** `main`-Funktion in `src/active_learning.py` implementieren:
    1.  Altes Datenset laden (aus `--dataset_path`).
    2.  Kandidaten generieren (S2.4).
    3.  Queries auswählen (S2.5).
    4.  HF-Modell abfragen (S2.6) mit `BudgetTracker`.
    5.  Neues Datenset = altes Datenset + neue Daten (konkatenieren).
    6.  Neues Datenset unter neuem Namen (aus `--output_path`) speichern.

---

## 5. Paket: Phase 4 (Integration & Haupt-Loop)

### 5.1 [WP S2/R1/V1] Integration & Orchestrierung
* `[ ]` **Aufgabe 5.1.1 (Hauptskript erstellen):** `src/main_experiment.py` erstellen.
* `[ ]` **Aufgabe 5.1.2 (Config-Parameter):** `MAX_AL_ITERATIONS` (z.B. 10), `N_QUERIES_PER_ITERATION` (z.B. 100), `N_ENSEMBLE_MODELS` (z.B. 5) zu `config.py` hinzufügen.
* `[ ]` **Aufgabe 5.1.3 (Initialisierung):** "Iteration 0" implementieren:
    1.  `python src/data_builder.py --output_path data/D_v1.pkl --n_episodes 100` aufrufen → `D_v1.pkl`.
    2.  `python src/fit_scaler.py --dataset_path data/D_v1.pkl --output_dir models/ --version 1` aufrufen → `scaler_state_v1.pkl`, `scaler_action_v1.pkl`.
    3.  Für `i in range(N_ENSEMBLE_MODELS)`: `python src/train_surrogate.py --dataset_path data/D_v1.pkl --state_scaler_path models/scaler_state_v1.pkl --action_scaler_path models/scaler_action_v1.pkl --output_dir models/ --version 1 --ensemble_index {i}` aufrufen → `surrogate_v1_ens_0.pth`, ..., `ens_4.pth`.
    4.  `python src/train_rl.py --surrogate_path models/surrogate_v1_ens_0.pth --state_scaler_path models/scaler_state_v1.pkl --action_scaler_path models/scaler_action_v1.pkl --save_path models/policy_v1.zip` aufrufen → `policy_v1.zip`.
    5.  `python src/evaluate_policy.py --policy_path models/policy_v1.zip --policy_type rl --n_episodes 20 --use_budget True` aufrufen → `baseline_reward_v1`.
    6.  Performance mit `ExperimentLogger` (1.5.2) loggen: Iteration=0, HF-Queries=initial_count, Reward=mean±std.
* `[ ]` **Aufgabe 5.1.4 (Implementierung des Haupt-Loops):** `for i in range(1, MAX_AL_ITERATIONS + 1):` Loop implementieren.
* `[ ]` **Aufgabe 5.1.5 (Loop-Schritt 1: Active Learning):** `python src/active_learning.py` aufrufen.
    * `[ ]` Input-Argumente:
        - `--dataset_path data/D_v{i-1}.pkl`
        - `--policy_path models/policy_v{i-1}.zip`
        - `--ensemble_dir models/` (lädt alle `surrogate_v{i-1}_ens_*.pth`)
        - `--state_scaler_path models/scaler_state_v{i-1}.pkl`
        - `--action_scaler_path models/scaler_action_v{i-1}.pkl`
        - `--n_queries N_QUERIES_PER_ITERATION`
        - `--output_path data/D_v{i}.pkl`
    * `[ ]` Budget-Check (1.3.4) erfolgt in `active_learning.py` (S2.6). Wenn Budget erschöpft: Script gibt Fehler zurück und Loop bricht ab.
* `[ ]` **Aufgabe 5.1.6 (Loop-Schritt 2: Scaler-Refit - KRITISCH):**
    * `[ ]` `python src/fit_scaler.py --dataset_path data/D_v{i}.pkl --output_dir models/ --version {i}` aufrufen.
    * `[ ]` Dies fittet **BEIDE** Scaler (state_scaler UND action_scaler) neu auf dem **gesamten erweiterten** Datenset.
    * `[ ]` Speichere als `scaler_state_v{i}.pkl` und `scaler_action_v{i}.pkl`.
* `[ ]` **Aufgabe 5.1.7 (Loop-Schritt 3: Surrogat-Retraining):**
    * `[ ]` Für `j in range(N_ENSEMBLE_MODELS)`: `python src/train_surrogate.py --dataset_path data/D_v{i}.pkl --state_scaler_path models/scaler_state_v{i}.pkl --action_scaler_path models/scaler_action_v{i}.pkl --output_dir models/ --version {i} --ensemble_index {j}` aufrufen.
    * `[ ]` Output: `surrogate_v{i}_ens_0.pth`, ..., `ens_4.pth`.
* `[ ]` **Aufgabe 5.1.8 (Loop-Schritt 4: RL-Retraining):**
    * `[ ]` `python src/train_rl.py --surrogate_path models/surrogate_v{i}_ens_0.pth --state_scaler_path models/scaler_state_v{i}.pkl --action_scaler_path models/scaler_action_v{i}.pkl --save_path models/policy_v{i}.zip` aufrufen.
    * `[ ]` **WICHTIG:** Nutzt die **neuen** Scaler `v{i}`, nicht die alten `v{i-1}`.
    * `[ ]` Output: `policy_v{i}.zip`.
* `[ ]` **Aufgabe 5.1.9 (Loop-Schritt 5: HF-Validierung & Logging):**
    * `[ ]` `python src/evaluate_policy.py --policy_path models/policy_v{i}.zip --policy_type rl --n_episodes 20 --use_budget True` aufrufen.
    * `[ ]` Ergebnisse mit `ExperimentLogger` loggen: Iteration={i}, Total_HF_Queries=..., Reward=mean±std.
* `[ ]` **Aufgabe 5.1.10 (Budget-Check & Loop-Stop):** 
    * `[ ]` Am Anfang jeder Iteration (5.1.4) den `BudgetTracker` (1.3.4) laden und prüfen: `if not budget_tracker.check_budget(min_required_time): break`.
    * `[ ]` Falls in 5.1.5 (Active Learning) ein Budget-Fehler zurückkommt: Loop abbrechen und finale Ergebnisse speichern.

---

## 6. Paket: Phase 5 (Analyse & Wissenschaftliche Methodik)

### 6.1 [WP A3] Baseline-Implementierung (Der Vergleichsmaßstab)
* `[ ]` **Aufgabe A3.1 (Baseline-Strategie definieren):** Einfache Baseline definieren (z.B. Statische Aktionen: `action = [0.5, 0.1]` konstant, oder Regel-basiert: "Wenn S < S_min, dS_add=1.0").
* `[ ]` **Aufgabe A3.2 (Baseline-Policy-Funktion):** Baseline-Logik in `src/evaluate_policy.py` als `baseline_policy_predict(state)` implementieren.
    * `[ ]` Wenn `--policy_type baseline`: nutze diese Funktion statt RL-Policy.
* `[ ]` **Aufgabe A3.3 (Baseline-Evaluierung):** 
    * `[ ]` `python src/evaluate_policy.py --policy_type baseline --n_episodes 20 --use_budget False` aufrufen.
    * `[ ]` **WICHTIG:** `use_budget=False`, da die Baseline-Evaluierung **separat** vom 8h-Haupt-Budget erfolgt (entweder vor dem Experiment oder als zusätzliche Analyse danach).
* `[ ]` **Aufgabe A3.4 (Baseline-Reporting):** Mean-Reward und Std-Reward der Baseline (A3.3) berechnen, speichern (z.B. `results/baseline_eval.json`) und in finale Plots (A5.3) als horizontale Linie eintragen.

### 6.2 [WP A1] Ablationsstudien (Wissenschaftliche Validierung)
* `[ ]` **Aufgabe A1.1 (Ablation: Random Sampling):** `src/main_experiment.py` (5.1.1) kopieren zu `src/main_ablation_random.py`.
* `[ ]` **Aufgabe A1.2 (Modifikation):** In `src/active_learning.py` eine Funktion `select_random_queries(candidates, n_queries)` implementieren (wählt zufällig).
    * `[ ]` `main_ablation_random.py` ruft `active_learning.py` mit Flag `--selection_method random` auf, das diese Funktion nutzt.
* `[ ]` **Aufgabe A1.3 (Ablations-Lauf):** `python src/main_ablation_random.py` mit identischem HF-Budget (8h) und Iterationen ausführen.
* `[ ]` **Aufgabe A1.4 (Ablation: Nur Initial-Daten):** "Iteration 0"-Performance (5.1.3) aus den Logs (1.5.2) extrahieren (bereits vorhanden, keine zusätzliche Berechnung nötig).

### 6.3 [WP A4] Hyperparameter-Tuning (HPT)
* `[ ]` **Aufgabe A4.1 (HPT-Skript):** `src/tune_surrogate.py` erstellen (z.B. mit `optuna`).
* `[ ]` **Aufgabe A4.2 (HPT-Ziel):** Surrogat-Hyperparameter (LR, HIDDEN_LAYERS aus 1.2.5) auf initialem Datenset (H0.8) optimieren, um `val_loss` (S1.5) zu minimieren.
* `[ ]` **Aufgabe A4.3 (Update Config):** Beste Hyperparameter (A4.2) in `config.py` (1.2.5) übertragen und **VOR** den finalen Experimenten (5.1.1, A1.3) nutzen.

### 6.4 [WP V2] Robustheitsanalyse
* `[ ]` **Aufgabe V2.1 (Modifikation):** `src/evaluate_policy.py` (V1.1) erweitern, um CLI-Argument `--param_shift` zu akzeptieren (z.B. `--param_shift MU_MAX=1.2`).
    * `[ ]` Lädt normale `params` aus Config und multipliziert die angegebene Variable mit dem Faktor.
* `[ ]` **Aufgabe V2.2 (Parameter-Shift):** 2-3 "Shift"-Szenarien definieren (z.B. `MU_MAX*1.2`, `K_S*0.8`, `YXS*1.1`).
* `[ ]` **Aufgabe V2.3 (Robustheits-Läufe):** 
    * `[ ]` Für jedes Shift-Szenario: `python src/evaluate_policy.py --policy_path models/policy_v{final}.zip --policy_type rl --n_episodes 20 --use_budget False --param_shift {scenario}` aufrufen.
    * `[ ]` **WICHTIG:** `use_budget=False`, da Robustheitsanalyse **zusätzlich** zum Haupt-Experiment erfolgt (keine Budget-Begrenzung).
* `[ ]` **Aufgabe V2.4 (Reporting):** Performance-Abfall berichten (z.B. "Reward unter +20% MU_MAX: 130.0 ± 8.5 (von 145.3 ± 10.2)"). Speichern in `results/robustness_eval.json`.

### 6.5 [WP A5] Finale Visualisierung & Multi-Objektiv-Analyse
* `[ ]` **Aufgabe A5.1 (Visualisierungs-Skript):** `src/visualize_results.py` erstellen.
    * `[ ]` Skript akzeptiert CLI-Argumente: `--results_dir`, `--output_dir`.
* `[ ]` **Aufgabe A5.2 (Daten laden):** Geloggte Ergebnisse aus `ExperimentLogger` (1.5.2) für:
    - Haupt-Lauf (`main_experiment.py`, 5.1.9)
    - Ablations-Lauf (`main_ablation_random.py`, A1.3)
    - Baseline (A3.4)
* `[ ]` **Aufgabe A5.3 (Plot 1: Der "Money Plot"):** Erstelle Linien-Plot mit `matplotlib` oder `seaborn`:
    * X-Achse: "Anzahl genutzter HF-Queries" (kumulative Summe über Iterationen).
    * Y-Achse: "Mean HF-Reward" (mit Error-Bars für Std).
    * Linie 1 (blau): Active Learning (Haupt-Lauf).
    * Linie 2 (orange): Random Sampling (Ablations-Lauf).
    * Horizontale Linie 3 (rot gestrichelt): Baseline-Heuristik (A3.4).
    * Speichere als `output_dir/money_plot.png`.
* `[ ]` **Aufgabe A5.4 (Plot 2: Multi-Objektiv-Analyse):** Erstelle Bar-Plot (gruppiert):
    * X-Achse: Ziel-Metriken ("Finale Biomasse", "Gesamte Substratkosten", "Episodenlänge").
    * Y-Achse: Wert der Metrik.
    * Balken-Gruppe 1: Beste RL-Policy (z.B. `policy_v{final}`).
    * Balken-Gruppe 2: Baseline-Heuristik (A3.4).
    * Zeige die **einzelnen** Ziele separat (extrahiere aus den geloggten Episoden die Komponenten: finale Biomasse aus `state[-1][biomasse_idx]`, Substratkosten aus `sum(actions[:, substrat_idx])`).
    * Speichere als `output_dir/multi_objective_plot.png`.
* `[ ]` **Aufgabe A5.5 (Plot 3: Surrogat-Genauigkeit):** Erstelle Linien-Plot:
    * X-Achse: "Anzahl genutzter HF-Queries".
    * Y-Achse: "Surrogat Validierungs-MSE" (geloggt in 5.1.7 mit `ExperimentLogger`).
    * Zeige, wie sich die Surrogat-Genauigkeit über die Iterationen verbessert.
    * Speichere als `output_dir/surrogate_accuracy_plot.png`.

### 6.6 [WP D1] Dokumentation
* `[ ]` **Aufgabe D1.1 (README erstellen):** `README.md` im Hauptverzeichnis erstellen.
* `[ ]` **Aufgabe D1.2 (Installation):** Setup (Paket 1.1) dokumentieren:
    ```bash
    conda env create -f environment.yml
    conda activate surrogate-ferment
    ```
* `[ ]` **Aufgabe D1.3 (Ausführung):** Reihenfolge dokumentieren:
    1.  **Haupt-Experiment:** `python src/main_experiment.py` (nutzt 8h-Budget, loggt zu WandB/TensorBoard).
    2.  **Ablations-Experiment:** `python src/main_ablation_random.py` (nutzt separates 8h-Budget).
    3.  **Baseline-Evaluierung:** `python src/evaluate_policy.py --policy_type baseline --use_budget False` (kein Budget).
    4.  **Robustheit (optional):** `python src/evaluate_policy.py --policy_path models/policy_v10.zip --param_shift MU_MAX=1.2 --use_budget False`.
    5.  **Visualisierung:** `python src/visualize_results.py --results_dir results/ --output_dir plots/`.
* `[ ]` **Aufgabe D1.4 (Konfiguration):** Dokumentiere die wichtigsten Config-Parameter (Budget, Iterationen, Hyperparameter) und wie man sie anpasst.
* `[ ]` **Aufgabe D1.5 (Troubleshooting):** Häufige Probleme dokumentieren (z.B. "Budget erschöpft vor Ende", "Surrogat-Modell divergiert", "RL-Training instabil").

---

## Ende der Aufgabenliste

**Zusammenfassung der Version 3 Verbesserungen:**
1. ✅ **Logging-Setup hinzugefügt** (1.5.1-1.5.3)
2. ✅ **CLI-Parameter für alle Skripte** dokumentiert
3. ✅ **Budget-Strategie geklärt** (Baseline/Robustheit nutzen KEIN Budget)
4. ✅ **Scaler-Refit präzisiert** (BEIDE Scaler neu fitten in 5.1.6)
5. ✅ **Budget-Tracking-Verantwortlichkeit** geklärt (VOR/NACH Episoden, nicht in solve_ivp)
6. ✅ **Ensemble-Pfade explizit** in 5.1.5
7. ✅ **Scaler-Pfade explizit** in S2.4
8. ✅ **Explizite CLI-Aufrufe** in 5.1.3 (Schritt-für-Schritt)
9. ✅ **Aktions-Skalierung** nicht mehr "eventuell", sondern explizit definiert
10. ✅ **Reward-Gewichte** in Config (1.2.6)
11. ✅ **Alle Argumente dokumentiert** für reproduzierbare Ausführung

Diese Liste ist nun **vollständig ausführbar** ohne Unklarheiten.
