from pathlib import Path
import pickle

import torch
from sklearn.metrics import f1_score

from models.generator import Generator
from dataloader import GenomeDataset
from config import train as train_cf
from config import model as model_cf
from config import dataset as dt_cfg
from config import features
from utils.train_utils import get_mean_for_each_label


def classification_with_test_data(test_path: Path, cls):
    score = cls.score(x, y)
    return score


if __name__ == '__main__':
    test_data_path = Path('dataset/8_202_test.csv')
    train_data_path = Path('dataset/8_202_train.csv')
    validation_model_paths = [
        Path('SavedModels/2022-12-02 18_02_55_LogisticRegression_8_202.pickle'),
        Path('SavedModels/2022-12-02 18_02_58_nn_8_202.pickle'),
        Path('SavedModels/2022-12-02 18_03_02_cnn1d_8_202.pickle'),
        Path('SavedModels/2022-12-02 18_03_04_knn_8_202.pickle'),
        Path('SavedModels/2022-12-02 18_03_05_svc_8_202.pickle'),
    ]
    generator_paths = [
        Path("SavedModels/2022-12-02 18_21_53acgan_without_mean_std_8_202.pth"),
        Path("SavedModels/2022-12-02 18_25_32acgan_mean_std_8_202.pth"),
        Path("SavedModels/2022-12-02 18_31_38basic_gan_without_mean_std_8_202.pth"),
        Path("SavedModels/2022-12-02 18_33_10basic_gan_with_mean_std_8_202.pth"),
    ]

    train_dataset = GenomeDataset(dt_cfg, train_data_path, features.SELECTED_FEATURES_64_8_202)
    x_train = train_dataset.scaler.fit_transform(train_dataset.x)
    y_train = train_dataset.y
    label_means, label_stds, unique_labels = get_mean_for_each_label(x_train, y_train)

    val_dataset = GenomeDataset(dt_cfg, test_data_path, features.SELECTED_FEATURES_64_8_202)
    x = val_dataset.scaler.fit_transform(val_dataset.x)
    y = val_dataset.y

    generators = [
        Generator(
            in_features=train_cf.LATENT_SIZE,
            out_features=len(features.SELECTED_FEATURES_64_8_202),
            hidden_layers=model_cf.G_HIDDEN_LAYERS,
        ) for _ in range(4)
    ]
    use_mean_std = [False, True, False, True]
    train_data_len = len(x_train)
    number_of_data = [train_data_len * i for i in (2, 5, 20, 50, 100)]
    for gen_i, gen in enumerate(generators):
        gen.load_state_dict(torch.load(generator_paths[gen_i]))
        gen.eval()
        print("*" * 40, f'Generator: {generator_paths[gen_i]}', "*" * 40)
        for validation_model_path in validation_model_paths:
            '''1st load validation model'''
            val_cls = pickle.load(open(validation_model_path, 'rb'))

            print('*' * 30, str(validation_model_path), '*' * 30)
            print(classification_with_test_data(test_data_path, val_cls))
            for data_len in number_of_data:
                '''3rd generate some data'''
                noise = torch.randn(data_len, train_cf.LATENT_SIZE, dtype=torch.float32)
                fake_classes = torch.randint(0, 2, (data_len, 1), dtype=torch.float32)
                if use_mean_std[gen_i]:
                    for ii, l in enumerate(unique_labels):
                        noise[fake_classes.squeeze(1) == l] = (noise[fake_classes.squeeze(1) == l] + label_means[ii]) * \
                                                              label_stds[ii]
                else:
                    noise = noise + fake_classes

                with torch.no_grad():
                    generated_data = gen(noise)
                '''4th validate generated data'''
                generated_data = generated_data.cpu().numpy()
                classes = fake_classes.cpu().numpy()
                preds = val_cls.predict(generated_data)
                print(f"Validation on {data_len} generated samples", 'Acc:',
                      val_cls.score(generated_data, classes),
                      'F1-Score:', f1_score(classes, preds))
