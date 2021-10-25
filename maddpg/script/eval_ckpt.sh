conda deactivate
conda activate advisor
cd ..

# ADVISOR -403.53
python3 eval_ckpt.py --scenario simple_spread_n3_vis --pretrained_model /home/iliu3/advisor/maddpg/ckpt_plot2/paper_advisor_simple_spread_n3_vis_1.6_lr5e-3_Alpha0.01_2/agents_best.ckpt --seed 2
# IL + RL static - 423.4
python3 eval_ckpt.py --scenario simple_spread_n3_vis --pretrained_model /home/iliu3/advisor/maddpg/ckpt_plot2/paper_static_simple_spread_n3_vis_vis1.6_AA_lr5e-3_1/agents.ckpt --seed 2
# IL -413.28
python3 eval_ckpt.py --scenario simple_spread_n3_vis --pretrained_model /home/iliu3/advisor/maddpg/ckpt_plot2/paper_imitation_simple_spread_n3_vis_1.6_3/agents_best.ckpt --seed 2
# RL -451.13
python3 eval_ckpt.py --scenario simple_spread_n3_vis --pretrained_model /home/iliu3/advisor/maddpg/ckpt_plot2/paper_RL_simple_spread_n3_vis_1.6_2/agents.ckpt --seed 2