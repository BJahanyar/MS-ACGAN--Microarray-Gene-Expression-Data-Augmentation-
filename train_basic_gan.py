from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch import optim

import config as cf
from dataloader import GenomeDataset, create_dataloader
from models import Generator, BasicDiscriminator
from utils.plots import plot_loss_acc
from utils.train_utils import create_real_labels, create_fake_labels, save_model, get_mean_for_each_label
from utils.utils import current_time

if __name__ == '__main__':
    Path(cf.model.SAVE_PATH).mkdir(exist_ok=True)
    cf.train.FIG_FOLDER.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GenomeDataset(cf.dataset, data_path=cf.train.TRAIN_DATA_PATH,
                            sample_features=cf.features.SELECTED_FEATURES_64_8_202)
    dataset.x = dataset.scaler.fit_transform(dataset.x)
    dataloader = create_dataloader(dataset, cf.dataset.DataLoader)

    label_means, label_stds, unique_labels = get_mean_for_each_label(dataset.x, dataset.y)

    sample_data, sample_label = dataset[0]

    gen = Generator(
        in_features=cf.train.LATENT_SIZE,
        out_features=dataset.data_size,
        hidden_layers=cf.model.G_HIDDEN_LAYERS,
    ).to(device)

    dis = BasicDiscriminator(
        input_features=dataset.data_size,
        hidden_layers=cf.model.D_HIDDEN_LAYERS,
    ).to(device)

    d_optim = optim.SGD(
        dis.parameters(),
        lr=cf.train.DIS_LEARNING_RATE
    )
    g_optim = optim.SGD(
        gen.parameters(),
        lr=cf.train.GEN_LEARNING_RATE,
    )

    source_loss_function = nn.MSELoss()

    loss_labels = ['dis_loss', 'gen_loss']

    dloss_gloss_racc_facc = np.zeros((2, cf.train.EPOCHS))

    for epoch in range(cf.train.EPOCHS):
        dis_loss, gen_loss = 0, 0
        for i, (data, labels) in enumerate(dataloader):
            real_labels = create_real_labels(len(data), device=device)
            fake_labels = create_fake_labels(len(data), device=device)

            data, labels = data.to(device), labels.to(device)
            for param in dis.parameters():
                param.grad = None

            _source = dis(data)
            dis_real_loss = source_loss_function(_source, real_labels)
            dis_real_loss.backward()
            d_optim.step()

            noise = torch.randn(len(data), cf.train.LATENT_SIZE, device=device, dtype=torch.float32)
            fake_classes = torch.randint(0, cf.train.NUM_CLASSES, (len(data), 1), device=device, dtype=torch.float32)
            # noise = noise + fake_classes
            for ii, l in enumerate(unique_labels):
                noise[fake_classes.squeeze(1) == l] = (noise[fake_classes.squeeze(1) == l] + label_means[ii]) * \
                                                  label_stds[ii]
            fake_data = gen(noise)
            _source = dis(fake_data.detach())
            dis_fake_loss = source_loss_function(_source, fake_labels)
            dis_fake_loss.backward()
            d_optim.step()

            for param in gen.parameters():
                param.grad = None

            _source = dis(fake_data)
            generator_loss = source_loss_function(_source, real_labels)
            generator_loss.backward()
            g_optim.step()

            dis_loss += dis_real_loss.item() + dis_fake_loss.item()
            gen_loss += generator_loss.item()

        dis_loss /= len(dataloader)
        gen_loss /= len(dataloader)
        dloss_gloss_racc_facc[0, epoch] = dis_loss
        dloss_gloss_racc_facc[1, epoch] = gen_loss

        print(f"\rGen Loss: {gen_loss:.4f}, Dis Loss: {dis_loss:.4f}")

    plot_loss_acc(dloss_gloss_racc_facc[:2], loss_labels, cf.train.FIG_FOLDER / (current_time() + 'loss.png'))
    save_model(gen, Path(cf.model.SAVE_PATH, current_time() + cf.model.GEN_NAME))
    np.savetxt(current_time() + cf.model.GEN_NAME[:-4] + '.txt', dloss_gloss_racc_facc)
