from my_utils import parser, loader, model, trainer, util
from warnings import filterwarnings

if __name__ == '__main__':
    # Ignore Warning Messages
    filterwarnings('ignore')

    # Random Seeds For Reproducibility
    # util.fix_random_seed()

    # Parser
    my_parser = parser.Parser(mode='train')
    my_args = my_parser.parse_args()
    print('*', util.parsed_arguments_dict(my_args))

    # Loader (Train / Valid)
    my_loader = loader.Loader(my_args.dataset_dir, my_args.height, my_args.width,
                              my_args.batch_size, mean_std=my_args.mean_std, center_crop_size=my_args.center_crop_size,
                              minus_1_to_plus_1_rescale=my_args.minus_1_to_plus_1_rescale)
    my_train_loader = my_loader.get_train_loader()
    my_valid_loader = my_loader.get_valid_loader()
    my_test_loader = my_loader.get_test_loader()

    # Model
    my_model = model.model(my_args.network_name, my_loader.num_classes, pretrained=False)

    # Train and Validation
    warmup_epochs = my_args.lr_warmup_epochs
    my_trainer = trainer.Trainer(my_model, my_train_loader, my_args.lr, my_loader.num_classes,
                                 my_args.loss_func, my_args.gamma, my_args.lr_step, my_args.lr_step_gamma,
                                 my_args.epochs, warmup_epochs, my_args.clip,
                                 my_args.progress_bar, my_args.gpu_index)
    for cur_epoch in range(0, my_args.epochs):
        my_trainer.train(cur_epoch, my_train_loader, lr_warmup=True if cur_epoch < warmup_epochs else False)
        my_trainer.valid(cur_epoch, my_valid_loader)

    # Test
    if my_args.test:
        my_trainer.test(my_test_loader)

    # Log
    util.write_log(my_args, my_trainer)
