from torch.utils.data import DataLoader
from tqdm import tqdm
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset, cycle
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta

images_folder = "/home/anastasija/Datasets/FFHQ/images1024x1024"
# device = "cuda:1"
batch_size = 1024
num_steps = 300

accelerator = Accelerator(
    split_batches = True,
    mixed_precision = 'no',
    kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]
)


dataset = Dataset(folder=images_folder, image_size=128, augment_horizontal_flip=False)
dl = DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = False, num_workers = 0)

# cycle_dl = cycle(dl)
step = 0

dl = accelerator.prepare(dl)
cycle_dl = cycle(dl)

for i in tqdm(range(len(dl))):
    imgs = next(cycle_dl) # .to(accelerator.device)

    if i % 10 == 0:
        print("-" * 80)
        print(i)
        print(imgs.shape, imgs.dtype, imgs.min(), imgs.max())
    


