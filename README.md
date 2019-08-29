# Solution to Visual Wakeup Words Challenge'19 (first place). 

**Participants:** Song Han, Ji Lin, Kuan Wang, Tianzhe Wang, Zhanghao Wu (following alphabetical order)

**Contact:** jilin@mit.edu

## Instruction

We have converted our model to tflite format with uint8 quantization. Here we provide a script to evaluate the model with PyTorch data loader in `eval.py`. However, to keep consistent with TensorFlow preprocessing, we used the preprocessing function imported from tensorflow. The preprocessing we used is defined in `preprocess.py`.

With the script, the model can get `94.575%` top-1 accuracy on minival set of VWW.

The demo code on Raspberry Pi and Android is included in this repo. 



## Usage

Run:

```
python eval.py
```

## Citation
```
@article{cai2018proxylessnas,
  title={Proxylessnas: Direct neural architecture search on target task and hardware},
  author={Cai, Han and Zhu, Ligeng and Han, Song},
  journal={International Conference on Learning Representations (ICLR)},
  year={2019}
}
```
