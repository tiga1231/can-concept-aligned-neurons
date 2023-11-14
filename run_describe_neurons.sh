python describe_neurons.py \
    --clip_model="ViT-L/14" \
    --d_probe='imagenet_val' \
    --target_model='resnet34' \
    --target_layers='conv1,layer1,layer2,layer3,layer4' \
    --concept_set='data/wordnet_hierarchy.txt'

python describe_neurons.py \
    --clip_model="ViT-L/14" \
    --d_probe='imagenet_val' \
    --target_model='resnet50' \
    --target_layers='conv1,layer1,layer2,layer3,layer4' \
    --concept_set='data/wordnet_hierarchy.txt'

    


    # --target_model='resnet50' \
    # --concept_set='data/20k_and_imagenet.txt'
    # --concept_set='data/20k.txt'
    # --concept_set='data/nouns_and_adjectives.txt'
    # --pool_mode='None'
