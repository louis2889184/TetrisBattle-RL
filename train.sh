python main.py --env-name "tetris_single" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 \
--value-loss-coef 0.5 --num-processes 126 --num-steps 32 --num-mini-batch 32 --log-interval 1 \
--use-linear-lr-decay --entropy-coef 0.01 --gpu 1 --num-env-steps 1000000000