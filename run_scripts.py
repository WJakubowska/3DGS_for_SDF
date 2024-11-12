import subprocess
import os



objects = ["dtu_scan24", "dtu_scan37", "dtu_scan40", "dtu_scan55", "dtu_scan63", "dtu_scan65", "dtu_scan69", "dtu_scan83",
           "dtu_scan97", "dtu_scan105", "dtu_scan106", "dtu_scan110", "dtu_scan114", "dtu_scan118", "dtu_scan122" ]

for program in ['train.py --eval --opt_sdf 0.5 --eps_s0 1e-8 --beta 300', 'render.py', "metrics.py"]:
    if program =='train.py --eval --opt_sdf 0.5 --eps_s0 1e-8 --beta 300':
        parameters_list = []
        for obj in objects:
            parameters = (
                f"-s /mnt/data/permuto_sdf_data/data_DTU/{obj}/ "
                f"-m /workspace/3DGS_for_SDF/results/dtu/{obj}/ "
                f"--model_sdf_path /workspace/permuto_sdf/checkpoints/serial_train/full_dtu_{obj}_with_mask_True_robo3/200000/models/sdf_model.pt "
                f"--mesh_path /workspace/permuto_sdf/results/output_permuto_sdf_meshes/dtu/with_mask_True/{obj}.ply"
            )
            parameters_list.append(parameters)
    else :
        parameters_list = []
        for obj in objects:
            parameters = (
                f"-m /workspace/3DGS_for_SDF/results/AKTUALNE/{obj} "
            )
            parameters_list.append(parameters)



    for i, params in enumerate(parameters_list):

        command = f"python {program} {params}"
        print(command)

        if " -m /workspace/3DGS_for_SDF/results/AKTUALNE" in command:
            name = command.split(" -m /workspace/3DGS_for_SDF/results/AKTUALNE")[1].split()[0]
            name =  "AKTUALNE" + name 
    
        print(name)

        output_file = f"/workspace/3DGS_for_SDF/results/{name}/output.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)

        print(f"RUN: {i+1}: SAVE: {output_file}")
