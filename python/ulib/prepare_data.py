import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import pickle
from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
import os


def resize_and_convert(img, size, quality=100):

	img = trans_fn.resize(img, (size, size), Image.LANCZOS)
	img = trans_fn.center_crop(img, size)
	buffer = BytesIO()
	img.save(buffer, format='jpeg', quality=quality)
	val = buffer.getvalue()

	return val
	# return img

def resize_multiple(img, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), quality=100):
	imgs = []

	for size in sizes:
		imgs.append(resize_and_convert(img, size, quality))

	return imgs


def resize_worker(img_file, sizes):
	i, file, label = img_file
	img = Image.open(file)
	img = img.convert('RGB')
	out = resize_multiple(img, sizes=sizes)

	return i, out, label


def prepare(transaction, dataset, n_worker, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), with_label=False, data_ratio=1.0):
	resize_fn = partial(resize_worker, sizes=sizes)

	files = sorted(dataset.imgs, key=lambda x: x[0])
	N = int(len(files) * data_ratio)
	print(N)
	files = [(i, file, label) for i, (file, label) in enumerate(files) if i <= N]
	total = 0

	with multiprocessing.Pool(n_worker) as pool:
		for i, imgs, label in tqdm(pool.imap_unordered(resize_fn, files)):
			for size, img in zip(sizes, imgs):
				data = pickle.dumps((img, label))
				key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
				# transaction.put(key, img)
				transaction.put(key, data)
			#
			total += 1

		transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--out', type=str)
	parser.add_argument('--n_worker', type=int, default=8)
	parser.add_argument('--path', type=str)
	parser.add_argument('--data_ratio', type=float, default=1.0)
	args = parser.parse_args()

	imgset = datasets.ImageFolder(args.path)

	# with lmdb.open("_".join([args.out, str(args.data_ratio)]), map_size=1024 ** 4, readahead=False) as env:
	with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
		with env.begin(write=True) as txn:
			prepare(txn, imgset, args.n_worker, sizes=(8, 16, 32, 64, 128, 256, 512), data_ratio=args.data_ratio)