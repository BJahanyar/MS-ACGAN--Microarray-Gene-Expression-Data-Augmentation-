from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch import optim

from config import train as train_cf
from config import model as model_cf
from config import dataset as dt_cfg
from config import features
from models import Generator, ACDiscriminator
from dataloader import GenomeDataset, create_dataloader
from utils.train_utils import (create_real_labels, create_fake_labels, compute_acc, save_model, Train,
                               get_mean_for_each_label)
from utils.plots import plot_loss_acc
from utils.utils import current_time

if __name__ == '__main__':
    Path(model_cf.SAVE_PATH).mkdir(exist_ok=True)
    train_cf.FIG_FOLDER.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GenomeDataset(dt_cfg, data_path=train_cf.TRAIN_DATA_PATH,
                            sample_features=features.SELECTED_FEATURES_64_8_202)
    dataset.x = dataset.scaler.fit_transform(dataset.x)
    dataloader = create_dataloader(dataset, dt_cfg.DataLoader)
    sample_data, sample_label = dataset[0]

    '''Calculating mean of each label'''
    label_means, label_stds, unique_labels = get_mean_for_each_label(dataset.x, dataset.y)

    gen = Generator(
        in_features=train_cf.LATENT_SIZE,
        out_features=dataset.data_size,
        hidden_layers=model_cf.G_HIDDEN_LAYERS,
    ).to(device)

    dis = ACDiscriminator(
        # num_classes=dataset.number_of_classes,
        num_classes=1,
        input_features=dataset.data_size,
        hidden_layers=model_cf.D_HIDDEN_LAYERS,
    ).to(device)

    # ToDo: Apply weight normalization
    # ToDo: Balance classes

    d_optim = optim.SGD(
        dis.parameters(),
        lr=train_cf.DIS_LEARNING_RATE
    )
    g_optim = optim.SGD(
        gen.parameters(),
        lr=train_cf.GEN_LEARNING_RATE,
    )

    source_loss_function = nn.MSELoss()
    cls_loss_function = nn.BCELoss()

    trainer = Train(
        generator=gen,
        discriminator=dis,
        g_optim=g_optim,
        d_optim=d_optim,
        source_loss=source_loss_function,
        class_loss=cls_loss_function,
        device=device,
    )

    loss_labels = ['dis_loss', 'gen_loss']
    acc_labels = ['real_acc', 'fake_acc']
    dloss_gloss_racc_facc = np.zeros((4, train_cf.EPOCHS))

    for epoch in range(train_cf.EPOCHS):
        dis_loss, gen_loss = 0, 0
        real_acc, fake_acc = 0, 0
        for i, (data, labels) in enumerate(dataloader):
            real_labels = create_real_labels(len(data), device=device)
            fake_labels = create_fake_labels(len(data), device=device)

            data, labels = data.to(device), labels.to(device)
            ''' train discriminator  with real data '''
            trainer.discriminator_zero_grad()

            dis_real_loss, _, _class = trainer.train_dis_with_real_data(data, labels, real_labels)

            real_accuracy = compute_acc(_class, labels)
            ''' train discriminator with fake data '''
            noise = torch.randn(len(data), train_cf.LATENT_SIZE, device=device, dtype=torch.float32)
            fake_classes = torch.randint(0, train_cf.NUM_CLASSES, (len(data), 1), device=device, dtype=torch.float32)
            for ii, l in enumerate(unique_labels):
                noise[fake_classes.squeeze(1) == l] = (noise[fake_classes.squeeze(1) == l] + label_means[ii]) * \
                                                      label_stds[ii]
            # noise = noise + fake_classes

            dis_fake_loss, _, _class, fake_data = trainer.train_dis_with_fake_data(fake_labels, noise, fake_classes)
            fake_accuracy = compute_acc(_class, fake_classes)

            ''' train generator '''
            trainer.generator_zero_grad()

            generator_loss = trainer.train_generator(fake_data, real_labels, fake_classes)

            dis_loss += dis_real_loss.item() + dis_fake_loss.item()
            gen_loss += generator_loss.item()
            real_acc += real_accuracy.item()
            fake_acc += fake_accuracy.item()

        dis_loss /= len(dataloader)
        gen_loss /= len(dataloader)
        real_acc /= len(dataloader)
        fake_acc /= len(dataloader)
        dloss_gloss_racc_facc[0, epoch] = dis_loss
        dloss_gloss_racc_facc[1, epoch] = gen_loss
        dloss_gloss_racc_facc[2, epoch] = real_acc
        dloss_gloss_racc_facc[3, epoch] = fake_acc

        print(f"\rGen Loss: {gen_loss:.4f}, Dis Loss: {dis_loss:.4f}"
              f"\tReal Acc: {real_acc:.4f} Fake Acc: {fake_acc:.4f}")

    plot_loss_acc(dloss_gloss_racc_facc[:2], loss_labels, train_cf.FIG_FOLDER / (current_time() + 'loss.png'))
    plot_loss_acc(dloss_gloss_racc_facc[2:], acc_labels, train_cf.FIG_FOLDER / (current_time() + 'acc.png'))
    save_model(trainer.generator, Path(model_cf.SAVE_PATH, current_time() + model_cf.GEN_NAME))
    np.savetxt(current_time() + model_cf.GEN_NAME[:-4] + '.txt', dloss_gloss_racc_facc)
