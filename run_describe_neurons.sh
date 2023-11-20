# Common args
CLIP_MODEL='ViT-L/14'
D_PROBE='imagenet_val'
TARGET_LAYER='conv1,layer1,layer2,layer3,layer4'
CONCEPT_SET='data/wordnet_hierarchy.txt'
SIMILARITY_FN='soft_wpmi_2'

python describe_neurons.py \
    --target_model='resnet34' \
    --activation_dir='my_data/resnet34/' \
    --dir_out='my_data/resnet34' \
    --clip_model="$CLIP_MODEL" \
    --d_probe="$D_PROBE" \
    --target_layer="$TARGET_LAYER" \
    --similarity_fn="$SIMILARITY_FN" \
    --concept_set="$CONCEPT_SET"

python describe_neurons.py \
    --target_model='resnet50' \
    --activation_dir='my_data/resnet50/' \
    --dir_out='my_data/resnet50'\
    --clip_model="$CLIP_MODEL" \
    --d_probe="$D_PROBE" \
    --target_layer="$TARGET_LAYER" \
    --similarity_fn="$SIMILARITY_FN" \
    --concept_set="$CONCEPT_SET"

# custom splits
python describe_neurons.py \
    --target_model='resnet50' \
    --activation_dir='my_data/resnet50_split0/' \
    --dir_out='my_data/resnet50_split0'\
    --model_weight='model_checkpoints/custom_resnet50_random_split0.ckpt' \
    --clip_model="$CLIP_MODEL" \
    --d_probe="$D_PROBE" \
    --target_layer="$TARGET_LAYER" \
    --similarity_fn="$SIMILARITY_FN" \
    --concept_set="$CONCEPT_SET"

python describe_neurons.py \
    --target_model='resnet50' \
    --activation_dir='my_data/resnet50_split1/' \
    --dir_out='my_data/resnet50_split1'\
    --model_weight='model_checkpoints/custom_resnet50_random_split1.ckpt' \
    --clip_model="$CLIP_MODEL" \
    --d_probe="$D_PROBE" \
    --target_layer="$TARGET_LAYER" \
    --similarity_fn="$SIMILARITY_FN" \
    --concept_set="$CONCEPT_SET"
    


# --pool_mode='None'
