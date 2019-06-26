import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataset import VWWDataset
import tensorflow as tf
import numpy as np
from preprocess import TFPreprocessEval
from tqdm import tqdm

quantize = True
print("Quantize:", quantize)


def main():
    # create model
    tflite_model_path = 'model_quantized.tflite'
    interpreter = tf.contrib.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # get input & output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tf.logging.info('input details: {}'.format(input_details))
    tf.logging.info('output details: {}'.format(output_details))

    input_shape = input_details[0]['shape']

    val_loader = torch.utils.data.DataLoader(
        VWWDataset(
            split='minival',
            transform=transforms.Compose([
                # transforms.Resize((input_size, int(input_size * hw_ratio))),
                transforms.Lambda(TFPreprocessEval(32).tf_preprocess_transform),
                transforms.ToTensor()
            ])
        ),
        batch_size=32, shuffle=False,
        num_workers=32, pin_memory=True)

    n_correct = 0
    n_sample = 0
    with torch.no_grad():
        for i, (images, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            images = images.permute(0, 2, 3, 1)
            images_np = images.cpu().numpy()
            if quantize:
                images_np = (images_np * 127 + 128).astype(np.uint8)
            target_np = target.cpu().numpy()
            for _ in range(images_np.shape[0]):
                interpreter.set_tensor(
                    input_details[0]['index'], images_np[_].reshape(*input_shape))
                interpreter.invoke()
                output_data = interpreter.get_tensor(
                    output_details[0]['index'])
                this_gt = target_np[_]
                pred = np.argmax(output_data.reshape(-1))
                if pred == this_gt:
                    n_correct += 1
                n_sample += 1
                if n_sample % 100 == 0:
                    print(n_correct * 1. / n_sample)


if __name__ == '__main__':
    main()
