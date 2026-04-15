# CSA_project-

#  Sentinel v3 — AI-Powered Focus Guardian

Sentinel v3 is a smart productivity tool that blocks distracting websites and uses machine learning to decide how difficult it should be to unlock them.

Instead of simply blocking access, it challenges the user with small tasks. The difficulty adapts based on user behavior.

---

## Features

*  Blocks distracting websites (Instagram, YouTube, etc.)
*  Uses Machine Learning (Random Forest) to predict distraction risk
*  Adaptive challenges (easy, medium, hard)
*  Tracks user behavior and focus stats
*  Model retrains automatically

---

## Requirements

Install required libraries:

```bash
pip install scikit-learn joblib numpy
```

---

## How to Run

### Mac/Linux:

```bash
sudo python3 sentinel_v3.py
```

### Windows (Run as Administrator):

```bash
python sentinel_v3.py
```

> Note: Admin/sudo permission is required for actual website blocking.

---

## Files

* `sentinel_v3.py` → Main code
* `sentinel_data.csv` → Stores user data
* `sentinel_model.pkl` → ML model

---

## How It Works

1. Blocks selected websites
2. When user tries to unlock:

   * AI predicts risk level
   * Gives challenge based on risk
3. If correct → temporary unlock
4. Data is saved and model improves over time

---

## Challenge Types

* Typing words
* Math problems
* Memory test
* Reverse typing
* Number sequence

---

## Customization

You can edit in code:

* `BLOCKED_SITES`
* `UNLOCK_SECONDS`
* `RETRAIN_EVERY`

---

## Note

If not run as admin, the app will run in **simulation mode** (no real blocking).


---

## Project Goal

To reduce distractions and improve focus using AI-based behavior tracking.
