"""
Fast test runner for AEGIS — patches config in-memory for a quick smoke test.
Runs 1 scenario (random), 3 Monte Carlo runs, 5 GNN epochs, 5 DQN episodes.
Usage: python test_fast.py
"""
import sys

# ── Patch config before any imports ──────────────────────────────────────────
import config
config.SIMULATION_CONFIG['n_runs']        = 3
config.SIMULATION_CONFIG['max_timesteps'] = 15
config.GNN_CONFIG['epochs']               = 5
# ── Patch DQN episodes ───────────────────────────────────────────────────────
import ai_modules
_orig_train = ai_modules.RLDefenseAgent.train
def _fast_train(self, G, attacker_mode='random', beta=0.3, episodes=60, **kw):
    return _orig_train(self, G, attacker_mode=attacker_mode, beta=beta, episodes=5, **kw)
ai_modules.RLDefenseAgent.train = _fast_train

# ── Delete cached models so it retrains fresh ────────────────────────────────
import os, glob
for f in glob.glob('models/rl_agent*'):
    try: os.remove(f)
    except: pass

# ── Run single scenario ───────────────────────────────────────────────────────
print("=" * 60)
print("AEGIS FAST TEST — reduced params")
print(f"  n_runs={config.SIMULATION_CONFIG['n_runs']}  "
      f"max_timesteps={config.SIMULATION_CONFIG['max_timesteps']}  "
      f"gnn_epochs={config.GNN_CONFIG['epochs']}  dqn_episodes=5")
print("=" * 60)

sys.argv = ['main.py', '--scenario', 'random', '--no_report']
exec(open('main.py').read())