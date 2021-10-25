conda deactivate
conda activate advisor
cd ..

ENV=simple_spread_n3_vis
LR=5e-3
alpha=0.4
VIS_RANGE=1.6
for i in 1 2 3 4;
do
    EXP_NAME=Advisor_${ENV}_${i}
    python3.7 main_advisor.py --alpha ${alpha} --scenario ${ENV} --actor_lr ${LR} --critic_lr ${LR} --exp_name ${EXP_NAME} --fixed_lr --critic_type gcn_max --eval_freq 1000 --vis_range ${VIS_RANGE} --seed ${i} --num_eval_runs 200
done




