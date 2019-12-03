python main.py --env-name "tetris_single" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 \
--value-loss-coef 0.5 --num-processes 128 --num-steps 32 --num-mini-batch 4 --log-interval 1 \
--use-linear-lr-decay --entropy-coef 0.01 --gpu 0 --num-env-steps 1000000000 --log-dir 'logs' \
--postfix "IOL_holes.01" --num-skip-frames 4