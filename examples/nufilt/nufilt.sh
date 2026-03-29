source activate
conda activate newqiu
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/path/to/fusion_bench:$PYTHONPATH


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/nufilt \
        fabric.loggers.name=vit-b-32-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=nufilt/nufilt \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/nufilt \
        fabric.loggers.name=vit-b-32-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=nufilt/nufilt \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch32
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/nufilt \
        fabric.loggers.name=vit-b-32-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=nufilt/nufilt \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch32
done



for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/nufilt \
        fabric.loggers.name=vit-b-16-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=nufilt/nufilt \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/nufilt \
        fabric.loggers.name=vit-b-16-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=nufilt/nufilt \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-base-patch16
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/nufilt \
        fabric.loggers.name=vit-b-16-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=nufilt/nufilt \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-base-patch16
done





for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/nufilt \
        fabric.loggers.name=vit-l-14-TA8 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=nufilt/nufilt \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/nufilt \
        fabric.loggers.name=vit-l-14-TALL14 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=nufilt/nufilt \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir=outputs/nufilt \
        fabric.loggers.name=vit-l-14-TALL20 \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method=nufilt/nufilt \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
        taskpool.base_model=openai/clip-vit-large-patch14
done


