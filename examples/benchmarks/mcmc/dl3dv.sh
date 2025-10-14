SCENE_DIR="data/dl3dv"
RESULT_DIR="results/benchmark_mcmc_dl3dv"
SCENE_LIST="0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3"
RENDER_TRAJ_PATH="ellipse"

CAP_MAX=1000000
DATA_FACTOR=1

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc \
        --strategy.cap-max $CAP_MAX \
        --disable_viewer \
        --data_factor $DATA_FACTOR \
        --data_type dl3dv \
        --init_type random \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

done
