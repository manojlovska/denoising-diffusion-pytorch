from train_utils import Train

images_folder = "/home/anastasija/Datasets/FFHQ/images1024x1024"
results_folder = "./results"
project_name = "denoising-diffusion-pytorch"
milestone = None

trainer = Train(images_folder=images_folder,
                results_folder=results_folder,
                project_name=project_name,
                milestone=milestone)

trainer.train()