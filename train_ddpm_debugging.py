from train_utils import Train
import argparse

def make_parser():
    parser = argparse.ArgumentParser("Argument Parser")

    parser.add_argument("-id", "--gpu_id", type=str, default=None)

    return parser

if __name__ == "__main__":

    args = make_parser().parse_args()
    gpu_id = str(args.gpu_id)

    images_folder = "/home/anastasija/Datasets/FFHQ/images1024x1024"
    results_folder = f"./results-{gpu_id}"
    project_name = f"denoising-diffusion-pytorch-{gpu_id}"

    trainer = Train(images_folder=images_folder,
                    results_folder=results_folder,
                    project_name=project_name)

    trainer.train()