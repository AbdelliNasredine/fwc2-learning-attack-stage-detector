from itertools import product
import train_v4

if __name__ == '__main__':
    datasets = ['dapt20', 'mscad']
    h_dims = [16, 32, 64, 128, 256]
    enc_depths = [2, 3, 4]
    prj_depths = [1, 2]
    batch_sizes = [256]
    c_rates = [round(i * 0.1, 1) for i in range(1, 10)]
    c_both_v = [False]
    taus = [1]

    grid_search_params = list(
        product(datasets, h_dims, enc_depths, prj_depths, batch_sizes, c_rates, c_both_v, taus))

    for combination in grid_search_params:
        ds, h_dim, enc_depth, prj_depth, batch_size, c_rate, c_both_v, tau = combination
        train_v4.main_fn(
            ds_name=ds,
            encoder_hidden_dim=h_dim,
            n_encoder_layers=enc_depth,
            n_projection_layers=prj_depth,
            cp=c_rate,
            corrupt_both_views=c_both_v,
            tau=tau,
            batch_size=batch_size,
            plot_tsne=False,
            wandb_prj_name=f's2d2-detector-exp-grid-search-{ds}'
        )
