# Common args
CLIP_MODEL='ViT-L/14'
D_PROBE='imagenet_val'
TARGET_LAYER='conv1,layer1,layer2,layer3,layer4'
CONCEPT_SET='data/wordnet_hierarchy.txt'
SIMILARITY_FN='soft_wpmi_2'

# alternative args
# ViT
TARGET_LAYER_VIT='encoder.layers.encoder_layer_1,encoder.layers.encoder_layer_2,encoder.layers.encoder_layer_3,encoder.layers.encoder_layer_4,encoder.layers.encoder_layer_5,encoder.layers.encoder_layer_6,encoder.layers.encoder_layer_7,encoder.layers.encoder_layer_8,encoder.layers.encoder_layer_9,encoder.layers.encoder_layer_10,encoder.layers.encoder_layer_11'

# Attack
D_PROBE_ATTACK='imagenet_val_attack'



# python describe_neurons.py \
#     --target_model='resnet34' \
#     --activation_dir='my_data/resnet34/' \
#     --dir_out='my_data/resnet34' \
#     --clip_model="$CLIP_MODEL" \
#     --d_probe="$D_PROBE" \
#     --target_layer="$TARGET_LAYER" \
#     --similarity_fn="$SIMILARITY_FN" \
#     --concept_set="$CONCEPT_SET"

# python describe_neurons.py \
#     --target_model='resnet50' \
#     --activation_dir='my_data/resnet50/' \
#     --dir_out='my_data/resnet50'\
#     --clip_model="$CLIP_MODEL" \
#     --d_probe="$D_PROBE" \
#     --target_layer="$TARGET_LAYER" \
#     --similarity_fn="$SIMILARITY_FN" \
#     --concept_set="$CONCEPT_SET"

# # custom splits
# python describe_neurons.py \
#     --target_model='resnet50' \
#     --activation_dir='my_data/resnet50_split0/' \
#     --dir_out='my_data/resnet50_split0'\
#     --model_weight='model_checkpoints/custom_resnet50_random_split0.ckpt' \
#     --clip_model="$CLIP_MODEL" \
#     --d_probe="$D_PROBE" \
#     --target_layer="$TARGET_LAYER" \
#     --similarity_fn="$SIMILARITY_FN" \
#     --concept_set="$CONCEPT_SET"

# python describe_neurons.py \
#     --target_model='resnet50' \
#     --activation_dir='my_data/resnet50_split1/' \
#     --dir_out='my_data/resnet50_split1'\
#     --model_weight='model_checkpoints/custom_resnet50_random_split1.ckpt' \
#     --clip_model="$CLIP_MODEL" \
#     --d_probe="$D_PROBE" \
#     --target_layer="$TARGET_LAYER" \
#     --similarity_fn="$SIMILARITY_FN" \
#     --concept_set="$CONCEPT_SET"

# # artificial
# python describe_neurons.py \
#     --target_model='resnet50' \
#     --activation_dir='my_data/resnet50_artificial/' \
#     --dir_out='my_data/resnet50_artificial/' \
#     --model_weight='model_checkpoints/resnet50_imagenet_artificial(550).ckpt' \
#     --clip_model="$CLIP_MODEL" \
#     --d_probe="$D_PROBE" \
#     --target_layer="$TARGET_LAYER" \
#     --similarity_fn="$SIMILARITY_FN" \
#     --concept_set="$CONCEPT_SET"

# # natural
# python describe_neurons.py \
#     --target_model='resnet50' \
#     --activation_dir='my_data/resnet50_natural/' \
#     --dir_out='my_data/resnet50_natural/' \
#     --model_weight='model_checkpoints/resnet50_imagenet_natural(450).ckpt' \
#     --clip_model="$CLIP_MODEL" \
#     --d_probe="$D_PROBE" \
#     --target_layer="$TARGET_LAYER" \
#     --similarity_fn="$SIMILARITY_FN" \
#     --concept_set="$CONCEPT_SET"

# # ViT
# python describe_neurons.py \
#     --target_model='vit_b_16' \
#     --activation_dir='my_data/vit_b_16/' \
#     --dir_out='my_data/vit_b_16/' \
#     --clip_model="$CLIP_MODEL" \
#     --d_probe="$D_PROBE" \
#     --target_layer="$TARGET_LAYER_VIT" \
#     --similarity_fn="$SIMILARITY_FN" \
#     --concept_set="$CONCEPT_SET"

# ## robust model
#  python describe_neurons.py \
#      --target_model='resnet50robust' \
#      --activation_dir='my_data/resnet50robust/' \
#      --dir_out='my_data/resnet50robust' \
#      --clip_model="$CLIP_MODEL" \
#      --d_probe="$D_PROBE" \
#      --target_layer="$TARGET_LAYER" \
#      --similarity_fn="$SIMILARITY_FN" \
#      --concept_set="$CONCEPT_SET"

# attacks
python describe_neurons.py \
    --target_model='resnet50' \
    --activation_dir='my_data/resnet50_under_attack/' \
    --dir_out='my_data/resnet50_under_attack' \
    --clip_model="$CLIP_MODEL" \
    --d_probe="$D_PROBE_ATTACK" \
    --target_layer="$TARGET_LAYER" \
    --similarity_fn="$SIMILARITY_FN" \
    --concept_set="$CONCEPT_SET"

python describe_neurons.py \
    --target_model='resnet50robust' \
    --activation_dir='my_data/resnet50robust_under_attack/' \
    --dir_out='my_data/resnet50robust_under_attack' \
    --clip_model="$CLIP_MODEL" \
    --d_probe="$D_PROBE_ATTACK" \
    --target_layer="$TARGET_LAYER" \
    --similarity_fn="$SIMILARITY_FN" \
    --concept_set="$CONCEPT_SET"

# python describe_neurons.py \
#     --target_model='resnet34' \
#     --activation_dir='my_data/resnet34_under_attack/' \
#     --dir_out='my_data/resnet34_under_attack' \
#     --clip_model="$CLIP_MODEL" \
#     --d_probe="$D_PROBE_ATTACK" \
#     --target_layer="$TARGET_LAYER" \
#     --similarity_fn="$SIMILARITY_FN" \
#     --concept_set="$CONCEPT_SET"

# --pool_mode='None'
