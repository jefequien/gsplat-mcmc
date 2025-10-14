SCENE_DIR="data/real-estate-10k"
RESULT_DIR="results/benchmark_mcmc_realestate10k"
# SCENE_LIST="0a0a998c176713fd"
# SCENE_LIST="0a0b1ce8a70f8c0c"
SCENE_LIST="0a0b7d56d8bb16f3"
RENDER_TRAJ_PATH="interp"

CAP_MAX=250000
DATA_FACTOR=1

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc \
        --strategy.cap-max $CAP_MAX \
        --data_factor $DATA_FACTOR \
        --data_type realestate10k \
        --init_type random \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

        # --disable_viewer \

done
