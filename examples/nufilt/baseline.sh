# Directory
# - line numbers below refer to this file
# - grouped by task setting -> model -> experiment
#
# 8 tasks: line 139
#   clip-vit-base-patch32: line 142
#     Weight Average: line 144
#     Continual Task Arithmetic: line 167
#     Continual Ties Merging: line 192
#     MagMax: line 217
#     Consensus TA: line 240
#     OPCM: line 263
#     C-Adamerging: line 286
#     C-WEMOE: line 309
#     WUDI: line 332
#     TSVM: line 355
#     KNOTS: line 378
#     ISO: line 403
#   clip-vit-base-patch16: line 426
#     Weight Average: line 428
#     Continual Task Arithmetic: line 451
#     Continual Ties Merging: line 476
#     MagMax: line 501
#     Consensus TA: line 524
#     OPCM: line 547
#     C-Adamerging: line 570
#     C-WEMOE: line 593
#     WUDI: line 616
#     TSVM: line 639
#     KNOTS: line 662
#     ISO: line 687
#   clip-vit-large-patch14: line 710
#     Weight Average: line 712
#     Continual Task Arithmetic: line 735
#     Continual Ties Merging: line 760
#     MagMax: line 785
#     Consensus TA: line 808
#     OPCM: line 831
#     C-Adamerging: line 854
#     C-WEMOE: line 877
#     WUDI: line 900
#     TSVM: line 923
#     KNOTS: line 946
#     ISO: line 971
# 14 tasks: line 995
#   clip-vit-base-patch32: line 998
#     Weight Average: line 1000
#     Continual Task Arithmetic: line 1023
#     Continual Ties Merging: line 1048
#     MagMax: line 1073
#     Consensus TA: line 1096
#     OPCM: line 1119
#     C-Adamerging: line 1142
#     C-WEMOE: line 1165
#     WUDI: line 1188
#     TSVM: line 1211
#     KNOTS: line 1234
#     ISO: line 1257
#   clip-vit-base-patch16: line 1280
#     Weight Average: line 1282
#     Continual Task Arithmetic: line 1305
#     Continual Ties Merging: line 1330
#     MagMax: line 1355
#     Consensus TA: line 1378
#     OPCM: line 1401
#     C-Adamerging: line 1424
#     C-WEMOE: line 1447
#     WUDI: line 1470
#     TSVM: line 1493
#     KNOTS: line 1516
#     ISO: line 1539
#   clip-vit-large-patch14: line 1562
#     Weight Average: line 1564
#     Continual Task Arithmetic: line 1587
#     Continual Ties Merging: line 1612
#     MagMax: line 1637
#     Consensus TA: line 1660
#     OPCM: line 1683
#     C-Adamerging: line 1706
#     C-WEMOE: line 1729
#     WUDI: line 1752
#     TSVM: line 1775
#     KNOTS: line 1798
#     ISO: line 1821
# 20 tasks: line 1845
#   clip-vit-base-patch32: line 1848
#     Weight Average: line 1850
#     Continual Task Arithmetic: line 1873
#     Continual Ties Merging: line 1898
#     MagMax: line 1923
#     Consensus TA: line 1946
#     OPCM: line 1969
#     C-Adamerging: line 1992
#     C-WEMOE: line 2015
#     WUDI: line 2038
#     TSVM: line 2061
#     KNOTS: line 2084
#     ISO: line 2107
#   clip-vit-base-patch16: line 2130
#     Weight Average: line 2132
#     Continual Task Arithmetic: line 2155
#     Continual Ties Merging: line 2180
#     MagMax: line 2205
#     Consensus TA: line 2228
#     OPCM: line 2251
#     C-Adamerging: line 2274
#     C-WEMOE: line 2297
#     WUDI: line 2320
#     TSVM: line 2343
#     KNOTS: line 2366
#     ISO: line 2389
#   clip-vit-large-patch14: line 2412
#     Weight Average: line 2414
#     Continual Task Arithmetic: line 2437
#     Continual Ties Merging: line 2462
#     MagMax: line 2487
#     Consensus TA: line 2510
#     OPCM: line 2533
#     C-Adamerging: line 2556
#     C-WEMOE: line 2579
#     WUDI: line 2602
#     TSVM: line 2625
#     KNOTS: line 2648
#     ISO: line 2671

source activate
conda activate newqiu
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/fusion_bench:$PYTHONPATH

# Shared settings:
# - versions: 0..9
# - seed: 42 + version
# - method.shuffle_order=true
# - method.save_on_every_step=false
# - method.evaluate_on_every_step=true

#######################################################################################################################################
# 8 tasks
#######################################################################################################################################

# clip-vit-base-patch32

# Weight Average
ROOT_DIR="outputs/weight_average"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/weight_average"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Task Arithmetic
ROOT_DIR="outputs/continual_task_arithmetic"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/task_arithmetic"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"
SCALING_FACTOR="0.3"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Ties Merging
ROOT_DIR="outputs/continual_ties_merging"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/ties_merging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"
SCALING_FACTOR="0.3"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# MagMax
ROOT_DIR="outputs/magmax"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/magmax"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Consensus TA
ROOT_DIR="outputs/consensus_ta"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/consensus_ta"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# OPCM
ROOT_DIR="outputs/opcm"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/opcm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-Adamerging
ROOT_DIR="outputs/c_adamerging"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/c_adamerging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-WEMOE
ROOT_DIR="outputs/c_wemoe"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/c_wemoe"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# WUDI
ROOT_DIR="outputs/wudi"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/wudi"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# TSVM
ROOT_DIR="outputs/tsvm"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/tsvm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# KNOTS
ROOT_DIR="outputs/knots"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/knots"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"
SCALING_FACTOR="0.3"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# ISO
ROOT_DIR="outputs/iso"
LOGGER_NAME="vit-b-32-TA8"
METHOD_NAME="nufilt/iso"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# clip-vit-base-patch16

# Weight Average
ROOT_DIR="outputs/weight_average"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/weight_average"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Task Arithmetic
ROOT_DIR="outputs/continual_task_arithmetic"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/task_arithmetic"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"
SCALING_FACTOR="0.3"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Ties Merging
ROOT_DIR="outputs/continual_ties_merging"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/ties_merging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"
SCALING_FACTOR="0.3"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# MagMax
ROOT_DIR="outputs/magmax"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/magmax"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Consensus TA
ROOT_DIR="outputs/consensus_ta"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/consensus_ta"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# OPCM
ROOT_DIR="outputs/opcm"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/opcm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-Adamerging
ROOT_DIR="outputs/c_adamerging"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/c_adamerging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-WEMOE
ROOT_DIR="outputs/c_wemoe"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/c_wemoe"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# WUDI
ROOT_DIR="outputs/wudi"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/wudi"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# TSVM
ROOT_DIR="outputs/tsvm"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/tsvm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# KNOTS
ROOT_DIR="outputs/knots"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/knots"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"
SCALING_FACTOR="0.3"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# ISO
ROOT_DIR="outputs/iso"
LOGGER_NAME="vit-b-16-TA8"
METHOD_NAME="nufilt/iso"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# clip-vit-large-patch14

# Weight Average
ROOT_DIR="outputs/weight_average"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/weight_average"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Task Arithmetic
ROOT_DIR="outputs/continual_task_arithmetic"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/task_arithmetic"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"
SCALING_FACTOR="0.3"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Ties Merging
ROOT_DIR="outputs/continual_ties_merging"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/ties_merging"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"
SCALING_FACTOR="0.3"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# MagMax
ROOT_DIR="outputs/magmax"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/magmax"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Consensus TA
ROOT_DIR="outputs/consensus_ta"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/consensus_ta"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# OPCM
ROOT_DIR="outputs/opcm"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/opcm"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-Adamerging
ROOT_DIR="outputs/c_adamerging"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/c_adamerging"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-WEMOE
ROOT_DIR="outputs/c_wemoe"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/c_wemoe"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# WUDI
ROOT_DIR="outputs/wudi"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/wudi"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# TSVM
ROOT_DIR="outputs/tsvm"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/tsvm"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# KNOTS
ROOT_DIR="outputs/knots"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/knots"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"
SCALING_FACTOR="0.3"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# ISO
ROOT_DIR="outputs/iso"
LOGGER_NAME="vit-l-14-TA8"
METHOD_NAME="nufilt/iso"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TA8"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

#######################################################################################################################################
# 14 tasks
#######################################################################################################################################

# clip-vit-base-patch32

# Weight Average
ROOT_DIR="outputs/weight_average"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/weight_average"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Task Arithmetic
ROOT_DIR="outputs/continual_task_arithmetic"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/task_arithmetic"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Ties Merging
ROOT_DIR="outputs/continual_ties_merging"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/ties_merging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# MagMax
ROOT_DIR="outputs/magmax"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/magmax"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Consensus TA
ROOT_DIR="outputs/consensus_ta"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/consensus_ta"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# OPCM
ROOT_DIR="outputs/opcm"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/opcm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-Adamerging
ROOT_DIR="outputs/c_adamerging"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/c_adamerging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-WEMOE
ROOT_DIR="outputs/c_wemoe"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/c_wemoe"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# WUDI
ROOT_DIR="outputs/wudi"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/wudi"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# TSVM
ROOT_DIR="outputs/tsvm"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/tsvm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# KNOTS
ROOT_DIR="outputs/knots"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/knots"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# ISO
ROOT_DIR="outputs/iso"
LOGGER_NAME="vit-b-32-TALL14"
METHOD_NAME="nufilt/iso"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# clip-vit-base-patch16

# Weight Average
ROOT_DIR="outputs/weight_average"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/weight_average"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Task Arithmetic
ROOT_DIR="outputs/continual_task_arithmetic"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/task_arithmetic"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Ties Merging
ROOT_DIR="outputs/continual_ties_merging"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/ties_merging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# MagMax
ROOT_DIR="outputs/magmax"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/magmax"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Consensus TA
ROOT_DIR="outputs/consensus_ta"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/consensus_ta"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# OPCM
ROOT_DIR="outputs/opcm"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/opcm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-Adamerging
ROOT_DIR="outputs/c_adamerging"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/c_adamerging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-WEMOE
ROOT_DIR="outputs/c_wemoe"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/c_wemoe"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# WUDI
ROOT_DIR="outputs/wudi"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/wudi"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# TSVM
ROOT_DIR="outputs/tsvm"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/tsvm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# KNOTS
ROOT_DIR="outputs/knots"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/knots"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# ISO
ROOT_DIR="outputs/iso"
LOGGER_NAME="vit-b-16-TALL14"
METHOD_NAME="nufilt/iso"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# clip-vit-large-patch14

# Weight Average
ROOT_DIR="outputs/weight_average"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/weight_average"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Task Arithmetic
ROOT_DIR="outputs/continual_task_arithmetic"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/task_arithmetic"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Ties Merging
ROOT_DIR="outputs/continual_ties_merging"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/ties_merging"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# MagMax
ROOT_DIR="outputs/magmax"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/magmax"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Consensus TA
ROOT_DIR="outputs/consensus_ta"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/consensus_ta"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# OPCM
ROOT_DIR="outputs/opcm"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/opcm"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-Adamerging
ROOT_DIR="outputs/c_adamerging"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/c_adamerging"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-WEMOE
ROOT_DIR="outputs/c_wemoe"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/c_wemoe"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# WUDI
ROOT_DIR="outputs/wudi"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/wudi"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# TSVM
ROOT_DIR="outputs/tsvm"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/tsvm"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# KNOTS
ROOT_DIR="outputs/knots"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/knots"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# ISO
ROOT_DIR="outputs/iso"
LOGGER_NAME="vit-l-14-TALL14"
METHOD_NAME="nufilt/iso"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL14"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

#######################################################################################################################################
# 20 tasks
#######################################################################################################################################

# clip-vit-base-patch32

# Weight Average
ROOT_DIR="outputs/weight_average"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/weight_average"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Task Arithmetic
ROOT_DIR="outputs/continual_task_arithmetic"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/task_arithmetic"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Ties Merging
ROOT_DIR="outputs/continual_ties_merging"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/ties_merging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# MagMax
ROOT_DIR="outputs/magmax"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/magmax"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Consensus TA
ROOT_DIR="outputs/consensus_ta"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/consensus_ta"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# OPCM
ROOT_DIR="outputs/opcm"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/opcm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-Adamerging
ROOT_DIR="outputs/c_adamerging"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/c_adamerging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-WEMOE
ROOT_DIR="outputs/c_wemoe"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/c_wemoe"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# WUDI
ROOT_DIR="outputs/wudi"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/wudi"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# TSVM
ROOT_DIR="outputs/tsvm"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/tsvm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# KNOTS
ROOT_DIR="outputs/knots"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/knots"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# ISO
ROOT_DIR="outputs/iso"
LOGGER_NAME="vit-b-32-TALL20"
METHOD_NAME="nufilt/iso"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch32"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# clip-vit-base-patch16

# Weight Average
ROOT_DIR="outputs/weight_average"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/weight_average"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Task Arithmetic
ROOT_DIR="outputs/continual_task_arithmetic"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/task_arithmetic"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Ties Merging
ROOT_DIR="outputs/continual_ties_merging"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/ties_merging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# MagMax
ROOT_DIR="outputs/magmax"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/magmax"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Consensus TA
ROOT_DIR="outputs/consensus_ta"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/consensus_ta"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# OPCM
ROOT_DIR="outputs/opcm"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/opcm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-Adamerging
ROOT_DIR="outputs/c_adamerging"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/c_adamerging"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-WEMOE
ROOT_DIR="outputs/c_wemoe"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/c_wemoe"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# WUDI
ROOT_DIR="outputs/wudi"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/wudi"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# TSVM
ROOT_DIR="outputs/tsvm"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/tsvm"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# KNOTS
ROOT_DIR="outputs/knots"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/knots"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# ISO
ROOT_DIR="outputs/iso"
LOGGER_NAME="vit-b-16-TALL20"
METHOD_NAME="nufilt/iso"
MODELPOOL="CLIPVisionModelPool/clip-vit-base-patch16_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-base-patch16"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# clip-vit-large-patch14

# Weight Average
ROOT_DIR="outputs/weight_average"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/weight_average"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Task Arithmetic
ROOT_DIR="outputs/continual_task_arithmetic"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/task_arithmetic"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Continual Ties Merging
ROOT_DIR="outputs/continual_ties_merging"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/ties_merging"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"
SCALING_FACTOR="0.1"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.scaling_factor="$SCALING_FACTOR" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# MagMax
ROOT_DIR="outputs/magmax"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/magmax"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# Consensus TA
ROOT_DIR="outputs/consensus_ta"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/consensus_ta"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# OPCM
ROOT_DIR="outputs/opcm"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/opcm"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-Adamerging
ROOT_DIR="outputs/c_adamerging"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/c_adamerging"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# C-WEMOE
ROOT_DIR="outputs/c_wemoe"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/c_wemoe"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# WUDI
ROOT_DIR="outputs/wudi"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/wudi"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# TSVM
ROOT_DIR="outputs/tsvm"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/tsvm"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# KNOTS
ROOT_DIR="outputs/knots"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/knots"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done

# ISO
ROOT_DIR="outputs/iso"
LOGGER_NAME="vit-l-14-TALL20"
METHOD_NAME="nufilt/iso"
MODELPOOL="CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only"
TASKPOOL="CLIPVisionModelTaskPool/clip-vit-classification_TALL20"
BASE_MODEL="openai/clip-vit-large-patch14"

for version in {0..9}; do
    fusion_bench \
        fabric.loggers.root_dir="$ROOT_DIR" \
        fabric.loggers.name="$LOGGER_NAME" \
        fabric.loggers.version=${version} \
        method.seed="$((42 + version))" \
        method="$METHOD_NAME" \
        method.shuffle_order=true \
        method.save_on_every_step=false \
        method.evaluate_on_every_step=true \
        modelpool="$MODELPOOL" \
        taskpool="$TASKPOOL" \
        taskpool.base_model="$BASE_MODEL"
done
