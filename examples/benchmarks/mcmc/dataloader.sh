SCENE_DIR="../../RenderingVideoGeneration/outputs/open_dataset/dl3dv/training"
RESULT_DIR="results/benchmark_mcmc_dataloader_dl3dv"
SCENE_LIST=(
    "batch_000_item_000"
    "batch_000_item_001"
    "batch_000_item_002"
    "batch_000_item_003"
    "batch_000_item_004"
    "batch_000_item_005"
    "batch_000_item_006"
    "batch_000_item_007"
    "batch_001_item_000"
    "batch_001_item_001"
    "batch_001_item_002"
    "batch_001_item_003"
    "batch_001_item_004"
    "batch_001_item_005"
    "batch_001_item_006"
    "batch_001_item_007"
)

RENDER_TRAJ_PATH="interp"
CAP_MAX=1000000
DATA_FACTOR=1

for SCENE in "${SCENE_LIST[@]}";
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc \
        --strategy.cap-max $CAP_MAX \
        --disable_viewer \
        --max_steps 7000 \
        --data_factor $DATA_FACTOR \
        --data_type blender \
        --init_type random \
        --init_extent 10.0 \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done
