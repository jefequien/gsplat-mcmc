SCENE_DIR="data/iphone"
SCENE_LIST=(
    "apple"
    "backpack"
)

RESULT_DIR="results/benchmark_iphone_1M_dynamic"
RENDER_TRAJ_PATH="interp"
DATA_FACTOR=2
CAP_MAX=1000000

for SCENE in "${SCENE_LIST[@]}";
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --deformation_opt \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

done