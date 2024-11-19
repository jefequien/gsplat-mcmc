SCENE_DIR="data/360_v2"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room treehill flowers"
SCENE_LIST="garden"
RENDER_TRAJ_PATH="ellipse"

# 1M GSs
RESULT_DIR="results/benchmark_mcmc_1M_neural"
CAP_MAX=1000000

# # 2M GSs
# RESULT_DIR="results/benchmark_mcmc_2M_neural"
# CAP_MAX=2000000

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    # train and eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
        --strategy.cap-max $CAP_MAX \
        --neural_opt \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE_DIR/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/
done

# # Zip the compressed files and summarize the stats
# if command -v zip &> /dev/null
# then
#     echo "Zipping results"
#     python benchmarks/compression/summarize_stats.py --results_dir $RESULT_DIR --scenes $SCENE_LIST
# else
#     echo "zip command not found, skipping zipping"
# fi
