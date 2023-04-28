datapath=miad
datasets=('catenary_dropper' 'electrical_insulator' 'metal_welding' 'nut_and_bolt'
               'photovoltaic_module' 'wind_turbine' 'witness_mark')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python main.py \
--gpu 3 \
--seed 0 \
--log_group simplenet_miad \
--log_project MIADAD_Results \
--results_path results \
--run_name run \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 40 \
--embedding_size 256 \
--gan_epochs 4 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 8 \
--resize 256 \
--imagesize 224 "${dataset_flags[@]}" miad $datapath

