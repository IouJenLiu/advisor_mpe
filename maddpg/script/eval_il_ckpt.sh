conda deactivate
conda activate advisor
cd ..

# IL -413.28
python3 eval_ckpt.py --scenario simple_spread_n3_vis --pretrained_model trained_ckpt/il_simple_spread_n3/agents.ckpt --seed 2