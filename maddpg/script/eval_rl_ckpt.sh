conda deactivate
conda activate advisor
cd ..

# RL
python3 eval_ckpt.py --scenario simple_spread_n3_vis --pretrained_model trained_ckpt/rl_simple_spread_n3/agents.ckpt --seed 2