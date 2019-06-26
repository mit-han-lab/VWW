# Submission to Visual Wakeup Words Challenge

**Participants:** Song Han, Ji Lin, Kuan Wang, Tianzhe Wang, Zhanghao Wu (following alphabetical order)

**Contact:** jilin@mit.edu

## Instruction

We have converted our model to tflite format with uint8 quantization. Here we provide a script to evaluate the model with PyTorch data loader in `eval.py`. However, to keep consistent with TensorFlow preprocessing, we used the preprocessing function imported from tensorflow. The preprocessing we used is defined in `preprocess.py`.

With the script, the model can get `94.575%` top-1 accuracy on minival set of VWW.



## Usage

Run:

```
python eval.py
```

## Citation
```
@article{han2019design,
  title={Design Automation for Efficient Deep Learning Computing},
  author={Han, Song and Cai, Han and Zhu, Ligeng and Lin, Ji and Wang, Kuan and Liu, Zhijian and Lin, Yujun},
  journal={arXiv preprint arXiv:1904.10616},
  year={2019}
}
```
