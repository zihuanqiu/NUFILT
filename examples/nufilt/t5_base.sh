source activate
conda activate newqiu
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/fusion_bench:$PYTHONPATH

fusion_bench \
    fabric.loggers.root_dir=outputs/c_adamerging_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=nufilt/c_adamerging_nlp \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 



fusion_bench \
    fabric.loggers.root_dir=outputs/c_wemoe_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=nufilt/c_wemoe_nlp \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 




fusion_bench \
    fabric.loggers.root_dir=outputs/wudi_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=nufilt/wudi_nlp \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 


fusion_bench \
    fabric.loggers.root_dir=outputs/ta_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=nufilt/task_arithmetic_nlp \
    method.shuffle_order=false \
    method.scaling_factor=0.3 \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 


fusion_bench \
    fabric.loggers.root_dir=outputs/ties_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=nufilt/ties_merging_nlp \
    method.shuffle_order=false \
    method.scaling_factor=0.3 \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 


fusion_bench \
    fabric.loggers.root_dir=outputs/opcm_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=nufilt/opcm_nlp \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=true \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 


fusion_bench \
    fabric.loggers.root_dir=outputs/nufilt_nlp \
    fabric.loggers.name=t5-base \
    fabric.loggers.version=0 \
    method.seed=42 \
    method=nufilt/nufilt_nlp \
    method.shuffle_order=false \
    method.save_on_every_step=false \
    method.evaluate_on_every_step=false \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation 
