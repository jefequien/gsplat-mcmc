SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark_mcmc_360_v2"
SCENE_LIST=(
    "garden"
    "bicycle"
    "stump"
    "bonsai"
    "counter"
    "kitchen"
    "room"
    # "treehill"
    # "flowers"
)

RENDER_TRAJ_PATH="ellipse"
CAP_MAX=1000000

for SCENE in "${SCENE_LIST[@]}";
do
    echo "Running $SCENE"

    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc \
        --strategy.cap-max $CAP_MAX \
        --disable_viewer \
        --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

done
