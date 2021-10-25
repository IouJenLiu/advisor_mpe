conda deactivate
conda activate advisor
cd ..

ENV=simple_spread_n3_vis
VIS_RANGE=1.6
for i in 1 2 3 4;
do
    EXP_NAME=RL_${ENV}_${i}
    python3.7 main_vec.py --scenario ${ENV}  --seed ${i} --exp_name ${EXP_NAME} --critic_type gcn_max --eval_freq 1000 --vis_range ${VIS_RANGE} --seed ${i} --num_eval_runs 200
done




