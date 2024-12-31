from itertools import product
import train_v4

datasets = [('dapt20', 5)]
enc_depth = [2, 3, 4]
prj_depth = [1, 2]
h_dim = [64, 128, 256]
l_rates = [0.01, 0.001]
batch_sizes = [1024]
c_rates = [round(i * 0.1, 1) for i in range(1, 10)]
c_both_v = [False, True]
tau = [1]


grid_search_params = list(product(datasets, enc_depth, prj_depth, h_dim, l_rates, batch_sizes, c_rates, c_both_v, tau))

for combination in grid_search_params:
    ds, enc_depth, prj_depth, h_dim, l_rate, batch_size, c_rate, c_both_v, tau = combination
    train_v4.main_fn(
        ds_name=ds[0],
        num_stages=ds[1],
        encoder_hidden_dim=h_dim,
        n_encoder_layers=enc_depth,
        n_projection_layers=prj_depth,
        cp=c_rate,
        corrupt_both_views=c_both_v,
        tau=tau,
        learning_rate=l_rate,
        batch_size=batch_size,
        plot_tsne=False,
    )