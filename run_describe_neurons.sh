python describe_neurons.py \
    --d_probe='imagenet_val' \
    --target_layers='conv1,layer1,layer2,layer3,layer4' \
    --concept_set='data/20k_and_imagenet.txt'
    # --concept_set='data/20k.txt'
    # --concept_set='data/nouns_and_adjectives.txt'
    # --pool_mode='None'
