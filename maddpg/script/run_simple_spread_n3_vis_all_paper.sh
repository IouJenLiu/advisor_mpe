conda deactivate
conda activate advisor
cd ..

ENV=simple_spread_n3_vis
i=0
LR=5e-3
alpha=0.4
for VIS_RANGE in 0.8 1.2 1.6 2;
do
  for i in 1 2 3 4;
  do
      EXP_NAME=paper_advisor_${ENV}_${VIS_RANGE}_lr${LR}_Alpha${alpha}_${i}
      #python3.7 main_advisor.py --alpha ${alpha} --scenario ${ENV} --actor_lr ${LR} --critic_lr ${LR} --exp_name ${EXP_NAME} --fixed_lr --critic_type gcn_max --eval_freq 1000 --vis_range ${VIS_RANGE} --seed ${i} --num_eval_runs 200 --save_dir ./ckpt_plot2 &
  done
done


echo 'run Imitation Learning'
ENV=simple_spread_n3_vis
for VIS_RANGE in 0.8 1.6 2;
do
  for i in 0 1 2 3 4;
  do
    EXP_NAME=paper_imitation_${ENV}_${VIS_RANGE}_${i}
    #python3.7 main_imitation.py --scenario ${ENV} --seed ${i}  --exp_name ${EXP_NAME} --critic_type gcn_max --eval_freq 1000 --vis_range ${VIS_RANGE} --seed ${i} --num_eval_runs 200 --save_dir ./ckpt_plot2 &
  done
done


echo 'run RL'
ENV=simple_spread_n3_vis
for VIS_RANGE in 1.2;
do
  for i in 0 1 2 3 4;
  do
    EXP_NAME=paper_RL_${ENV}_${VIS_RANGE}_${i}
    python3.7 main_vec.py --scenario ${ENV}  --seed ${i} --exp_name ${EXP_NAME} --critic_type gcn_max --eval_freq 1000 --vis_range ${VIS_RANGE} --seed ${i} --num_eval_runs 200 --save_dir ./ckpt_plot2 &
  done
done

LR=5e-3
for VIS_RANGE in 0.8 1.2 1.6 2;
do
  for i in 1 2 3 4;
  do
      EXP_NAME=paper_static_${ENV}_vis${VIS_RANGE}_AA_lr${LR}_${i}
      #python3.7 main_advisor.py --use_static_advisor_weight --scenario ${ENV} --actor_lr ${LR} --critic_lr ${LR} --exp_name ${EXP_NAME} --fixed_lr --critic_type gcn_max --eval_freq 1000 --vis_range ${VIS_RANGE} --seed ${i} --num_eval_runs 200 --save_dir ./ckpt_plot2 &
  done
done

