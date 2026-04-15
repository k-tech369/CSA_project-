"""
SENTINEL v3 — AI-Powered Focus Guardian
========================================
Blocks distracting sites and uses ML to decide
how hard to make the unlock challenge based on
your past behaviour patterns.

Requirements:
    pip install scikit-learn joblib numpy
Run:
    sudo python3 sentinel_v3.py      (Mac/Linux)
    python sentinel_v3.py            (Windows, as Admin)
"""

import os, sys, csv, time, random, joblib
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# ──────────────────────────────────────────
# SETTINGS  (edit these freely)
# ──────────────────────────────────────────
UNLOCK_SECONDS  = 60          # how long sites stay unlocked
MIN_ROWS        = 8           # min data rows before ML activates
RETRAIN_EVERY   = 5           # retrain after every N new saves
DATA_FILE       = "sentinel_data.csv"
MODEL_FILE      = "sentinel_model.pkl"

BLOCKED_SITES = [
    "www.instagram.com", "instagram.com",
    "www.facebook.com",  "facebook.com",
    "www.twitter.com",   "twitter.com",
    "www.reddit.com",    "reddit.com",
    "www.youtube.com",   "youtube.com",
    "www.tiktok.com",    "tiktok.com",
]

HOSTS_PATH = (
    r"C:\Windows\System32\drivers\etc\hosts"
    if sys.platform.startswith("win")
    else "/etc/hosts"
)


# ──────────────────────────────────────────
# TERMINAL UI HELPERS
# ──────────────────────────────────────────
W = 52   # terminal width

def clear():    os.system("cls" if sys.platform.startswith("win") else "clear")
def line():     print("─" * W)
def dline():    print("═" * W)


def header():
    dline()
    print("  🛡️   S E N T I N E L   |   FOCUS GUARDIAN   v3")
    dline()

def info(msg):  print(f"  ℹ️   {msg}")
def ok(msg):    print(f"  ✅  {msg}")
def warn(msg):  print(f"  ⚠️   {msg}")
def err(msg):   print(f"  ❌  {msg}")

def countdown(seconds: int):
    for left in range(seconds, 0, -1):
        m, s = divmod(left, 60)
        print(f"\r  ⏱️   Re-locking in {m:02d}:{s:02d}", end="", flush=True)
        time.sleep(1)
    print()


# ──────────────────────────────────────────
# SITE BLOCKER
# ──────────────────────────────────────────
class SiteBlocker:
    REDIRECT = "127.0.0.1"
    simulating = False

    def _clean(self, lines: list[str]) -> list[str]:
        return [l for l in lines if not any(l.strip().endswith(s) for s in BLOCKED_SITES)]

    def lock(self):   self._apply(active=True)
    def unlock(self): self._apply(active=False)

    def _apply(self, active: bool):
        try:
            with open(HOSTS_PATH, "r") as f:
                   lines = f.readlines()
            with open(HOSTS_PATH, "w") as f:
                f.writelines(self._clean(lines))
                if active:
                    f.writelines(f"{self.REDIRECT} {s}\n" for s in BLOCKED_SITES)
            ok("Sites LOCKED 🔒" if active else "Sites UNLOCKED 🔓")
        except PermissionError:
            self.simulating = True
            status = "LOCKED" if active else "UNLOCKED"
            info(f"[SIMULATION] Would be {status}. Run as sudo/admin for real blocking.")


# ──────────────────────────────────────────
# CHALLENGES
# ──────────────────────────────────────────
def challenge_easy() -> Tuple[str, str]:
    """Type a focus word."""
    words = ["FOCUS", "GRIND", "WORK", "DEEP", "FLOW", "STUDY", "BUILD"]
    w = random.choice(words)
    return f"  Type the word:  {w}", w

def challenge_medium() -> Tuple[str, str]:
    """Solve a two-step arithmetic expression."""
    a, b, c = random.randint(2, 9), random.randint(2, 9), random.randint(2, 9)
    ans = a + b * c        # respects BODMAS
    return f"  Solve (BODMAS):  {a} + {b} × {c}", str(ans)

def challenge_hard_math() -> Tuple[str, str]:
    """Harder multi-operator arithmetic."""
    a, b, c = random.randint(5, 15), random.randint(5, 15), random.randint(2, 9)
    ans = a * b - c
    return f"  Solve:  ({a} × {b}) − {c}", str(ans)

def challenge_memory() -> Tuple[str, str]:
    """Memorise a number shown briefly, then type it back."""
    number = str(random.randint(1000, 99999))
    print(f"\n  Memorise this:  {number}")
    time.sleep(3)
    print("\r" + " " * 30 + "\r", end="")   # wipe the number
    return "  Type the number you saw:", number


def challenge_reverse() -> Tuple[str, str]:
    """Type a word backwards."""
    words = ["python", "focus", "sentinel", "keyboard", "monitor"]
    w = random.choice(words)
    return f"  Type this word BACKWARDS:  {w}", w[::-1]

def challenge_sequence() -> Tuple[str, str]:
    """Continue a simple arithmetic sequence."""
    start = random.randint(1, 10)
    step  = random.randint(2, 7)
    seq   = [start + step * i for i in range(4)]
    ans   = start + step * 4
    return f"  Next number in:  {seq} → ?", str(ans)

# Difficulty tiers
CHALLENGES = {
    0: [challenge_easy],
    1: [challenge_medium, challenge_reverse, challenge_sequence],
    2: [challenge_hard_math, challenge_memory],
}

RISK_LABELS = {
    0: ("✅ Low risk",    "Easy challenge"),
    1: ("🟡 Medium risk", "Medium challenge"),
    2: ("🔴 High risk",   "Hard challenge"),
}

def get_challenge(risk: int) -> Tuple[str, str]:
    fn = random.choice(CHALLENGES[risk])
    return fn()



# ──────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────
class DataStore:
    HEADER = ["hour", "weekday", "resp_time", "correct", "unlocks", "streak", "label"]

    def save(self, hour, weekday, resp_time, correct, unlocks, streak):
        label = self._label(hour, unlocks)
        exists = os.path.isfile(DATA_FILE)
        with open(DATA_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(self.HEADER)
            w.writerow([hour, weekday, resp_time, correct, unlocks, streak, label])
        return self.count()

    def load(self) -> Optional[np.ndarray]:
        if not os.path.exists(DATA_FILE):
            return None
        data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
        return data if len(data) >= MIN_ROWS else None

    def count(self) -> int:
        if not os.path.exists(DATA_FILE):
            return 0
        with open(DATA_FILE) as f:
            return sum(1 for _ in f) - 1

    @staticmethod
    def _label(hour: int, unlocks: int) -> int:
        late_night = hour >= 22 or hour < 6
        return int(unlocks > 5 or (late_night and unlocks > 2))


    def stats(self):
        data = self.load()
        if data is None:
            warn("Not enough data yet — keep using Sentinel!")
            return
        hours     = data[:, 0].astype(int)
        resp      = data[:, 2]
        correct   = data[:, 3]
        unlocks   = data[:, 4]
        labels    = data[:, -1]
        peak      = int(np.bincount(hours).argmax())
        line()
        print("  📊  YOUR FOCUS STATS")
        line()
        print(f"  Sessions logged       : {len(data)}")
        print(f"  Correct answer rate   : {correct.mean()*100:.1f}%")
        print(f"  Avg response time     : {resp.mean():.1f}s")
        print(f"  Total unlocks granted : {int(unlocks.sum())}")
        print(f"  High-risk sessions    : {labels.mean()*100:.1f}%")
        print(f"  Peak distraction hour : {peak:02d}:00")
        line()


# ──────────────────────────────────────────
# ML MODEL
# ──────────────────────────────────────────
class RiskModel:
    def __init__(self, store: DataStore):
        self.store = store
        self.model: Optional[Pipeline] = None
        self._load_or_train()

    def _build(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100, random_state=42)),
        ])

    def _load_or_train(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
                return
            except Exception:
                pass
        self.train()

    def train(self):
        data = self.store.load()
        if data is None:
            self.model = None
            return

        X, y = data[:, :-1], data[:, -1].astype(int)
        if len(np.unique(y)) < 2:
            self.model = None
            return

        m = self._build()
        m.fit(X, y)

        # show cross-val accuracy if enough data
        if len(data) >= 10:
            scores = cross_val_score(m, X, y, cv=min(3, len(data)), scoring="accuracy")
            info(f"Model accuracy: {scores.mean()*100:.1f}% (±{scores.std()*100:.1f}%)")

        joblib.dump(m, MODEL_FILE)
        self.model = m

    def predict_risk(self, hour, weekday, unlocks, streak) -> int:
        """Returns 0 (low), 1 (medium), or 2 (high) risk."""
        if self.model is None:
            # cold-start heuristic
            if hour >= 22 or hour < 6:
                return 2
            if unlocks > 3:
                return 1
            return 0

        features = np.array([[hour, weekday, 4.0, 1, unlocks, streak]])
        prob = self.model.predict_proba(features)[0][1]

        if prob < 0.35:   return 0
        if prob < 0.65:   return 1
        return 2


# ──────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────
def run():
    clear()
    header()

    blocker  = SiteBlocker()
    store    = DataStore()
    ml_model = RiskModel(store)

    unlocks  = 0
    streak   = 0
    rows_this_session = 0

    blocker.lock()

    while True:
        print(f"\n  Streak: {streak}🔥  |  Unlocks: {unlocks}")
        print("  [u] Unlock   [s] Stats   [q] Quit")
        choice = input("  → ").strip().lower()

        if choice == "q":
            blocker.unlock()
            print("\n  👋  Stay focused. See you!\n")
            break

        elif choice == "s":
            store.stats()

        elif choice == "u":
            now     = datetime.now()
            hour    = now.hour
            weekday = now.weekday()

            risk  = ml_model.predict_risk(hour, weekday, unlocks, streak)
            rlabel, clabel = RISK_LABELS[risk]
            line()
            print(f"  {rlabel}  →  {clabel}")

            prompt, answer = get_challenge(risk)
            print(prompt)

            start   = time.time()
            attempt = input("  >> ").strip()
            elapsed = round(time.time() - start, 2)

            correct = int(attempt == answer)

            # save & maybe retrain
            rows_this_session += 1
            total = store.save(hour, weekday, elapsed, correct, unlocks, streak)
            if total % RETRAIN_EVERY == 0:
                ml_model.train()
                info("Model updated with latest data.")

            if correct:
                streak  += 1
                unlocks += 1
                ok(f"Correct! ({elapsed}s)  Streak: {streak}🔥")
                blocker.unlock()
                countdown(UNLOCK_SECONDS)
                blocker.lock()
            else:
                streak = 0
                err(f"Wrong!  Answer was: {answer}")

        else:
            warn("Unknown command. Use u / s / q")


if __name__ == "__main__":
    run()