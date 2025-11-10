# SCENE_DIR="data/dl3dv-frames256"
# RESULT_DIR="results/benchmark_mcmc_dl3dv-frames256"
# SCENE_LIST=(
#     "2K/2aadacbd3623dff25834e1f13cb6c1d6f91996e2957e8fd7de1ca7883e424393"
#     "3K/5dac8fa15625e54b1bd487b36701fb99c8ed909563b86ce3728caebfefde8dda"
#     "3K/50208cdb39510fdf8dedbd57536a6869dc93027d77608983925c9d7955578882"
#     "5K/de3b622853799be8b8b5f2acd72c32f58e45b0b12f482c4b1f555a4602302424"
#     "5K/2799116f2a663fee45296028841c2f838e525246a5dcb67a2e6699b7f3deabed"
#     "8K/0a45d3a3470b042d8aa44340601b58d258ad123464c3ae8c2899882ab8cc5bfd"
#     "8K/41976b976a156d2744b3ff760ea1385b3d1dd53d002ecb1ac4b2953515d5c30e"
#     "9K/7f7e34027e51bd908dac27e0c4459180842031047b0f516f5a84ab5bc24c5f44"
#     "10K/970a5c674c27b504d592d0a70c496d0e35ab0dc76802fb6e1bf336a4c1fe150a"
#     "10K/98f8220e7d2a2addbacaf26d3b005764e0510ae730ac11182c6e4de15a7e02e4"
# )

# SCENE_DIR="data/dl3dv-evaluation-frames256"
# RESULT_DIR="results/benchmark_mcmc_dl3dv-evaluation-frames256"
SCENE_DIR="data/dl3dv-evaluation-frames256-long"
RESULT_DIR="results/benchmark_mcmc_dl3dv-evaluation-frames256-long"
SCENE_LIST=(
    "0ec879405fe1a7a393876fed993657e0f7dda7fa16788ad48c669e9b67215a3c"
    "1e5ce991775e9266dcd553306ae9cc153ffd19101b789ff66526bc3877c54fdf"
)

RENDER_TRAJ_PATH="interp"
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
        --data_type blender \
        --init_type random \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

done
