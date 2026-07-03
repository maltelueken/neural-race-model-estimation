#!/bin/bash

python scripts/c2st_recovery.py c2st.mode=hierarchical \
    c2st.recovery_dir="outputs/rdm/model.num_bins\=12/model.num_mid\=128/optimizer\=adam_cosine_decay/train_steps\=500000"