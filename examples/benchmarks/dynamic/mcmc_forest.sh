SCENE_DIR="data/forest_dataset"
SCENE_LIST=(
    "dyntree_IMG_0001_20230810_mlshih"
    "dyntree_IMG_0012_20230810_mlshih"
    "dyntree_IMG_0017_20230810_mlshih"
    "dyntree_IMG_0023_20230810_mlshih"
    "dyntree_IMG_1697_gaochen"
    "dyntree_IMG_1698_gaochen"
    "dyntree_IMG_1700_gaochen"
    "dyntree_IMG_1701_gaochen"
    "dyntree_IMG_1703_gaochen"
    "dyntree_IMG_1704_gaochen"
)

RESULT_DIR="results/benchmark_forest_dynamic_2_4M"
RENDER_TRAJ_PATH="interp"
DATA_FACTOR=2
CAP_MAX=4000000

for SCENE in "${SCENE_LIST[@]}";
do
    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # ffmpeg -y -framerate 30 -pattern_type glob -i "$RESULT_DIR/$SCENE/renders/train_step29999_*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$RESULT_DIR/$SCENE/renders/train_step29999.mp4"
    # ffmpeg -y -framerate 30 -pattern_type glob -i "$RESULT_DIR/$SCENE/renders/val_step29999_*.png" -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p "$RESULT_DIR/$SCENE/renders/val_step29999.mp4"

done
