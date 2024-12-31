#!/bin/bash

# Define parameter values
datasets=("dapt20")
enc_depth=(2 3 4)
prj_depth=(1 2)
h_dim=(64 128 256)
l_rates=(0.01 0.001)
batch_sizes=(1024)
c_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
c_both_v=(False True)
tau=(1)

for dataset in "${datasets[@]}"; do
    for enc in "${enc_depth[@]}"; do
        for prj in "${prj_depth[@]}"; do
            for h in "${h_dim[@]}"; do
                for lr in "${l_rates[@]}"; do
                    for bs in "${batch_sizes[@]}"; do
                        for cr in "${c_rates[@]}"; do
                            for cbv in "${c_both_v[@]}"; do
                                for t in "${tau[@]}"; do
                                    echo "RUN PARAMS:"
                                    echo "ds_name=$ds_name, encoder_hidden_dim=$h, n_encoder_layers=$enc, n_projection_layers=$prj, cp=$cr, corrupt_both_views=$cbv, tau=$t, learning_rate=$lr, batch_size=$bs, plot_tsne=False"

                                    python3 train_v4.py --ds $ds_name --edim $h --nel $enc --npl $prj --cr $cr --cr_both $cbv --
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
