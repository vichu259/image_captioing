#Image Captioning

Baseline - https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning


Requirements:

Python 2.x or Python3.x

PyTorch 0.4

CUDA 0.9 (optional)

nltk 3.2.2


Train:

python train.py
Test:  python test_with_beamsearch.py --img='png/example.png' --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --vocab_path='data/vocab.pkl' --beam_size=5

Classes Changed:

  bulid vocab - Using Andrej Karpathy's caption data. As it contains entire CocoDataset in one json.

Models:

  encoder - changed the cnn encoder to give features instead of one single combined vector which are used for attention.
  decoder - added attention class

test_with_beamsearch:

   added this class to test an image.
