import os
import numpy as np
from PIL import Image
from tqdm import tqdm

np.random.seed(42)

dirname = "<PATH_TO_IMG_DIR>" # Change here.
print(dirname)

images = []

for filename in tqdm(sorted(os.listdir(dirname))):
    pathname = os.path.join(dirname, filename)

    with Image.open(pathname) as image:
        image = np.asarray(image)
        images.append(image)

images = np.stack(images)

# We need to randomly shuffle the order due to a inception score calculation bug.
# https://github.com/openai/guided-diffusion/issues/165
p = np.random.permutation(len(images))
images = images[p]

np.savez_compressed(f"{dirname}.npz", arr_0=images)
