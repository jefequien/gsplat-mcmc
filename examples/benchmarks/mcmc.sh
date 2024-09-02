# SCENE_DIR="data/360_v2"
# RESULT_DIR="results/benchmark_mcmc_1M"
# SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
# RENDER_TRAJ_PATH="ellipse"

# SCENE_DIR="data/bilarf/bilarf_data/testscenes"
# RESULT_DIR="results/benchmark_bilarf"
# SCENE_LIST="chinesearch lionpavilion pondbike statue strat building"
# RENDER_TRAJ_PATH="spiral"

# SCENE_DIR="data/bilarf/bilarf_data/editscenes"
# RESULT_DIR="results/benchmark_bilarf"
# SCENE_LIST="rawnerf_windowlegovary rawnerf_sharpshadow scibldg"
# RENDER_TRAJ_PATH="spiral"

SCENE_DIR="data/shiny"
RESULT_DIR="results/benchmark_shiny"
SCENE_LIST="cd" # crest food giants lab pasta seasoning tools"
RENDER_TRAJ_PATH="spiral"

CAP_MAX=1000000

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=1
    fi

    echo "Running $SCENE"

    # train without eval
    # CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
    #     --strategy.cap-max $CAP_MAX \
    #     --render_traj_path $RENDER_TRAJ_PATH \
    #     --data_dir $SCENE_DIR/$SCENE/ \
    #     --result_dir $RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --disable_viewer --data_factor $DATA_FACTOR \
            --strategy.cap-max $CAP_MAX \
            --render_traj_path $RENDER_TRAJ_PATH \
            --data_dir $SCENE_DIR/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done
done


for SCENE in $SCENE_LIST;
do
    echo "=== Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/val*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done