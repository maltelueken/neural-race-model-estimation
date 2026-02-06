
# target_dir=outputs/rdm/model.num_bins\=12/model.num_mid\=128/optimizer\=adam_cosine_decay/train_steps\=200000/
target_dir=outputs/crdm/model.dt\=0.0005/model.num_bins\=12/model.num_mid\=128/optimizer\=adam_cosine_decay/train_steps\=500000/

mkdir -p ${target_dir}

scp -r mluken@snellius:/projects/prjs1372/racing-diffusion-conflict/${target_dir}/* ./${target_dir}/
