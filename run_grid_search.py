from lightning_sdk import Studio, Machine

studio = Studio()

studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']

learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]

grid_search_params = [(lr, bs) for lr in learning_rates for bs in batch_sizes]

for index, (lr, bs) in enumerate(grid_search_params):
    cmd = f'python finetune_sweep.py --lr {lr} --batch_size {bs} --max_steps {100}'
    job_name = f'run-2-exp-{index}'
    job_plugin.run(cmd, machine=Machine.A10G, name=job_name)
