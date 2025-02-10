from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import lightning as L
from src.utils import parse_args
from src.models.factory import model_factory

def main(args):
    data_paths = {'train': args.train_data_path, 'test': args.test_data_path}
    batch_sizes = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size}
    model, data_module = model_factory(args.architecture, data_paths, batch_sizes, args.num_workers, args.val_split)

    logger = CSVLogger("logs", name=args.model_name, version=args.version)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath="logs/{}/version_{}/checkpoints".format(args.model_name, args.version), filename=args.model_name + '-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}', save_top_k=1, mode='min')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')
    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator="gpu", devices=1, logger=logger, log_every_n_steps=50, callbacks=[checkpoint_callback, early_stopping_callback])
    trainer.fit(model, data_module)
    
    best_checkpoint_path = checkpoint_callback.best_model_path
    test_model = model.load_from_checkpoint(best_checkpoint_path, model=model)
    trainer.test(test_model, datamodule=data_module)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)