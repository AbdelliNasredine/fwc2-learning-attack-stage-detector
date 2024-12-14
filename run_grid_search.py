from lightning_sdk import Studio, Machine

# reference to the current studio
# if you run outside of Lightning, you can pass the Studio name
studio = Studio()

# use the jobs plugin
studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']

# do a sweep over learning rates
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]

# a grid search combines all params
grid_search_params = [(lr, bs) for lr in learning_rates for bs in batch_sizes]

# start all jobs on an A10G GPU with names containing an index
for index, (lr, bs) in enumerate(grid_search_params):
    cmd = f'python finetune_sweep.py --lr {lr} --batch_size {bs} --max_steps {100}'
    job_name = f'run-2-exp-{index}'
    job_plugin.run(cmd, machine=Machine.A10G, name=job_name)
