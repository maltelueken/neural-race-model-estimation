#!/bin/bash

python scripts/c2st_recovery.py c2st.mode=single \
    "c2st.single_recovery_dirs=[ \
    'multirun/rdm/model.num_bins=12/model.num_mid=128/optimizer=adam_cosine_decay/test_num_obs=50/train_steps=500000', \
    'multirun/rdm/model.num_bins=12/model.num_mid=128/optimizer=adam_cosine_decay/test_num_obs=250/train_steps=500000', \
    'multirun/rdm/model.num_bins=12/model.num_mid=128/optimizer=adam_cosine_decay/test_num_obs=500/train_steps=500000', \
    'multirun/rdm/model.num_bins=12/model.num_mid=128/optimizer=adam_cosine_decay/test_num_obs=1000/train_steps=500000']"
