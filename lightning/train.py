import pytorch_lightning as pl
from lightning.model import SinglePlausibleClassification
from lightning.data import SemEvalDataModule
import argparse


def cli_main():
    pl.seed_everything(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model_name_or_path', default='google/electra-large-discriminator')
    parser.add_argument('--num_labels', default=3, type=int)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = SinglePlausibleClassification.add_specific_args(parser)
    args = parser.parse_args()

    train_loader = SemEvalDataModule.train_dataloader()
    dev_loader = SemEvalDataModule.val_dataloader

    model = SinglePlausibleClassification(args)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, dev_loader)


if __name__ == '__main__':
    cli_main()
