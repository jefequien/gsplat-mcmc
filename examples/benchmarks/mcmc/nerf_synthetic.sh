SCENE_DIR="data/nerf_synthetic"
RESULT_DIR="results/benchmark_mcmc_nerf_synthetic"
SCENE_LIST=(
    "chair"
    "drums"
    "ficus"
    "hotdog"
    "lego"
    "materials"
    "mic"
    "ship"
)

RENDER_TRAJ_PATH="ellipse"
CAP_MAX=250000
DATA_FACTOR=1

for SCENE in "${SCENE_LIST[@]}";
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc \
        --strategy.cap-max $CAP_MAX \
        --disable_viewer \
        --data_factor $DATA_FACTOR \
        --data_type blender \
        --init_type random \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done
