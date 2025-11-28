SCENE_DIR="data/dl3dv-frames256"
RESULT_DIR="results/benchmark_mcmc_rae"
SCENE_LIST=(
    "0000"
    "0001"
    "0002"
    "0003"
    "0004"
    "0005"
    "0006"
    "0007"
    "0008"
    "0009"
)

RENDER_TRAJ_PATH="exact"
CAP_MAX=250000
DATA_FACTOR=1

for SCENE in "${SCENE_LIST[@]}";
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc \
        --strategy.cap-max $CAP_MAX \
        --disable_viewer \
        --max_steps 7000 \
        --data_factor $DATA_FACTOR \
        --data_type rae \
        --init_type random \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

done
