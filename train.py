from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import lightning as L
from src.utils import parse_args
from src.models.factory import model_factory

def main(args):
    L.seed_everything(42, workers=True)
    data_paths = {'train': args.train_data_path, 'test': args.test_data_path}
    batch_sizes = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size}
    model, data_module, lit_module = model_factory(args.architecture, data_paths, batch_sizes, args.num_workers, args.val_split, 
                                       args.learning_rate, image_size=128,
                                       number_patches=16, checkpoint_path_list=args.checkpoint_path_list, 
                                       pretrained_model_name_list=args.pretrained_model_name_list, 
                                       freeze=args.freeze, number_waveforms=args.num_waveforms, 
                                       signals_per_snr=args.signals_per_snr)

    logger = CSVLogger("logs", name=args.model_name, version=args.version)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', dirpath="logs/{}/version_{}/checkpoints".format(args.model_name, args.version), filename=args.model_name + '-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}', save_top_k=1, mode='max')
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=args.patience, mode='max')
    trainer = L.Trainer(max_epochs=args.max_epochs, accelerator="gpu", devices=1, logger=logger, log_every_n_steps=50, callbacks=[checkpoint_callback, early_stopping_callback], deterministic=True)
    
    if not args.test:
        trainer.fit(model, data_module)
        
        best_checkpoint_path = checkpoint_callback.best_model_path
        test_model = lit_module.load_from_checkpoint(best_checkpoint_path, model = model.model, number_waveforms=args.number_waveforms, signals_per_snr = args.signals_per_snr)
        trainer.test(test_model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)