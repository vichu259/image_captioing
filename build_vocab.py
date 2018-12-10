import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import json

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(jsons, threshold,max_len=100,min_word_freq=5):
    with open(jsons, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()
    vocab = Vocabulary()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        if img['split'] in {'train', 'restval'}:
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_captions.append(captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    vocab.add_word('<pad>')
    for i, word in enumerate(words):
        vocab.add_word(word)
    # word_map = {k: v + 1 for v, k in enumerate(words)}
    vocab.add_word('<unk>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    return vocab

def main(args):
    vocab = build_vocab(jsons=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/dataset_coco.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
