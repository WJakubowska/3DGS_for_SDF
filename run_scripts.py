import subprocess


program = './train.py'


parameters_list = [
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_0.1 --opt_sdf 0.1 --port 6011',
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_0.2 --opt_sdf 0.2 --port 6011',
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_0.3 --opt_sdf 0.3 --port 6011',
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_0.4 --opt_sdf 0.4 --port 6011',
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_0.5 --opt_sdf 0.5 --port 6011',
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_0.6 --opt_sdf 0.6 --port 6011',
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_0.7 --opt_sdf 0.7 --port 6011',
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_0.8 --opt_sdf 0.8 --port 6011',
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_0.9 --opt_sdf 0.9 --port 6011',
    '-s /mnt/data/permuto_sdf_data/nerf_synthetic/lego -w -m lego_loss_sdf_1 --opt_sdf 1.0 --port 6011',

]


for i, params in enumerate(parameters_list):

    command = f"python {program} {params}"
    
    output_file = f"output_{i+1}.txt"
    with open(output_file, 'w') as f:
        subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)

    print(f"RUN: {i+1}: SAVE: {output_file}")
