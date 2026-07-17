# Curriculum-based deep RL modeling of multi-attribute decision-making in macaques (CRAM)

Project starter code — a three-part, fill-in-the-blank [marimo](https://marimo.io)
project that adapts a **resource-rational meta-MDP** to a macaque
**multiattribute-choice + eye-tracking** task. Start from
raw gaze, discover how the animals sample information, turn that into a model, and fit
the model to individual monkeys' behavior.

## How to use this starter code

After the lecture, you can work through the parts at your own pace. 
If you complete all three, **you'll have a scaffold for
implementing this as a project.**

Recommended use: take the tools and analyses each part
introduces and **re-implement the steps yourself, in your own coding style** — rebuild
the marimo notebooks your own way. The `solutions/` copies are there to check yourself
against.

Each notebook has `TODO` blanks. Fill in the marked functions; a reactive
self-check turns **green (✅)** when your implementation is right and stays a grey
*"not done yet"* until then. Everything downstream re-runs automatically, so work
**top to bottom** — later exercises reuse the functions you wrote earlier.

## Install

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, pandas, scipy, matplotlib, marimo
```

## Open a notebook

```bash
marimo edit Notebooks/01_within_option_looking.py
```

Run each notebook from anywhere in the repo — the notebooks locate the project root
themselves (they look for the folder containing both `Data/` and `Libraries/`).

## The parts

| Part | Description |
|---|---|
| `00_meta_mdp.py` | Reading part: the meta-MDP model and how it maps onto the macaque task. |
| `01_within_option_looking.py` | Investigation of the raw eye-tracking data. |
| `02_attribute_level_model.py` | Model modification to accomodate the finding. |
| `03_individual_differences.py` | Fit a per-monkey parameter so the model reproduces individual differences in monkeys gaze patterns. |

## Data 
You can download this separately [here](https://www.dropbox.com/scl/fo/s35lkrar4n3lcuk092iyw/AB7A5UFvaiOIIboGJOhi_I4?rlkey=4bdjwewg7ue50bjg7e8yv53ye&dl=0). 

## Layout

```
CRAM-rl-starter/
├── README.md                 ← this file
├── REFERENCE.md              ← data schema + library-file reference
├── requirements.txt
├── Notebooks/
│   ├── 00_meta_mdp.py
│   ├── 01_within_option_looking.py
│   ├── 02_attribute_level_model.py
│   ├── 03_individual_differences.py
│   ├── assets/               ← figures used by Part 1
│   └── solutions/            ← answer key (one filled-in copy per part)
├── Libraries/                ← the model code the notebooks import
│   ├── data.py
│   ├── metamdp_nhp.py
│   ├── train_reinforce.py
│   └── task_perkins.py
├── Data/
│   ├── all_trial_options.csv ← per-trial options table
│   └── example_traces.npz    ← 500 Hz eye traces for two example sessions
└── Papers/                   ← background reading (PDFs)
    ├── PerkinsRich_PLOSBio.pdf
    ├── Perkins_JoCN.pdf
    └── Radulescu_OPMI.pdf
```

For the data schema and what each library file does, see [REFERENCE.md](REFERENCE.md).

## Papers

Background reading in `Papers/` — the task and the model this starter code adapts:

- **`Perkins_JoCN.pdf`** — Perkins et al. (Journal of Cognitive Neuroscience): The source of the task and eye-tracking data used here.
- **`PerkinsRich_PLOSBio.pdf`** — Perkins et al. (PLOS Biology): *"Orbitofrontal cortex
  computes gaze-dependent comparisons between attributes rather than integrated
  values."*  Paper detailing the neural findings from the task.
- **`Radulescu_OPMI.pdf`** — Radulescu et al. (Open Mind): *"A Resource-Rational Account of Human
  Eye Movements During Immersive Visual Search."* The resource-rational meta-MDP that the
  starter code adapts to this task.
