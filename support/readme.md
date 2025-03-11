Files `tinyimagenet_refined_labels.json` and `imagenet_refined_labels.json` are used to specify train/val splits when we re-implement adversarial finetuning (AFT) methods. We re-implemented three AFT methods (TeCoA, PMG-AFT and FARE) in the paper on the tinyImageNet dataset. For each class, we randomly sample 10% of the instances in the training set for evaluation in the phase of finetuning. In the `.json` file, each class (identified with a synset id) has the following attributes:
```
{
  synset identifier (e.g., 'n01443537'):
  {
    "clean_name": cleansed textual name (e.g., 'goldfish'),
    "wordnet_def": definition given by WordNet (e.g., 'small golden or orange-red freshwater fishes of Eurasia used as pond or aquarium fishes'),
    "eval_files": list of instances selected as evaluation data (e.g., ["n01443537_266.JPEG", "n01443537_75.JPEG", ...])
  },
}
```

We also provide the cleansed textual name for each class and its definition given by [WordNet](https://wordnet.princeton.edu/).
