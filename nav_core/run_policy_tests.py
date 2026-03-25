from pathlib import Path

try:
    from nav_core.PolicyRunner import PolicyRunner
except ModuleNotFoundError:
    from PolicyRunner import PolicyRunner

MODEL_PATH = Path(__file__).resolve().parent / "models" / "finalmodel.zip"
runner = PolicyRunner(str(MODEL_PATH))

test_cases = [
    ("goal right",  [100.0, 100.0, 400.0, 100.0]),
    ("goal left",   [400.0, 100.0, 100.0, 100.0]),
    ("goal down",   [100.0, 100.0, 100.0, 400.0]),
    ("goal up",     [100.0, 400.0, 100.0, 100.0]),
    ("diag down-right", [100.0, 100.0, 400.0, 400.0]),
    ("at goal",     [300.0, 300.0, 300.0, 300.0]),
]

for name, values in test_cases:
    action = runner.predict_action(*values)
    print(name, "->", action, "(", runner.action_name(action), ")")
