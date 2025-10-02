SCENE_DIR="data/iphone"
SCENE_LIST=(
    # "apple"
    # "block"
    # "paper-windmill"
    # "space-out"
    # "spin"
    "teddy"
    # "wheel"
)

RESULT_DIR="results/benchmark_iphone_4M_dynamic"
RENDER_TRAJ_PATH="interp"
DATA_FACTOR=1
CAP_MAX=4000000

for SCENE in "${SCENE_LIST[@]}";
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --data_type dycheck \
        --init_type random \
        --init_num_pts 400000 \
        --sh_degree 0 \
        --deformation_opt \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

done