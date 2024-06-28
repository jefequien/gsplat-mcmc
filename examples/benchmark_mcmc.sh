RESULT_DIR=results/mcmc_random

for SCENE in bonsai counter kitchen room stump bicycle garden;
do
    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "stump" ] || [ "$SCENE" = "garden" ]; then
        DATA_FACTOR=4
    else
        DATA_FACTOR=2
    fi

    if [ "$SCENE" = "bicycle" ] || [ "$SCENE" = "stump" ] || [ "$SCENE" = "garden" ]; then
        CAP_MAX=3000000
    elif [ "$SCENE" = "bonsai" ]; then
        CAP_MAX=1218809
    elif [ "$SCENE" = "counter" ]; then
        CAP_MAX=1186522
    elif [ "$SCENE" = "kitchen" ]; then
        CAP_MAX=1792551
    elif [ "$SCENE" = "room" ]; then
        CAP_MAX=1581834
    else
        CAP_MAX=1000000
    fi

    echo "Running $SCENE"

    # train without eval
    python simple_trainer_mcmc.py --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --init_type random \
        --cap_max $CAP_MAX \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        python simple_trainer.py --disable_viewer --data_factor $DATA_FACTOR \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT
    done
done


for SCENE in bicycle bonsai counter garden kitchen room stump;
do
    echo "=== Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/val*;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done