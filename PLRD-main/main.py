import os
import wandb
from data import get_datamodule
from discovery import DiscoveryModelManager
from argparse import ArgumentParser
from Utils.memory import Memory
from Utils.util import set_seed
from pretrain import PretrainModelManager


def main(args):
    # Classes quantity division
    # banking
    if args.n_ind_class == 47:  # 40%
        args.n_ood_classes = [10, 10, 10]
    elif args.n_ind_class == 32:  # 60%
        args.n_ood_classes = [15, 15, 15]
    elif args.n_ind_class == 17:  # 80%
        args.n_ood_classes = [20, 20, 20]
    # clinc
    elif args.n_ind_class == 90:  # 40%
        args.n_ood_classes = [20, 20, 20]
    elif args.n_ind_class == 60:  # 60%
        args.n_ood_classes = [30, 30, 30]
    elif args.n_ind_class == 30:  # 80%
        args.n_ood_classes = [40, 40, 40]
    else:
        raise ValueError()

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.max_stage = len(args.n_ood_classes)  # Number of stages of continuous discovery
    data = get_datamodule(args)
    data.prepare_data()
    memory = Memory(args)

    # IND pre-training
    if args.pretrain:
        args.stage = 0
        data.setup(stage=args.stage)
        args.n_train_example = data.get_n_train_example(args.stage)  # Number of training samples
        print('Pre-training begin...')
        # If using the ind pre training weights downloaded from huggingface, annotate following pre-training code
        # p_manager = PretrainModelManager(args)
        # p_manager.train(args, data.train_dataloader(), data.val_dataloader())
        # p_manager.test(args, data.test_dataloader())
        if args.use_memory:
            memory.select_ind_exemplars_to_store(data.train_dataloader())  # Store IND exemplars
            memory.print_memory_info()
        print('Pre-training finished!')

    # Continual OOD discovery
    if args.train:
        print('Training begin...')
        model_name = 'pretrain_' + args.dataset + '_' + str(args.n_ind_class) + '_divide_seed_' + str(
            args.divide_seed) + '.bin'
        pretrain_model_path = os.path.join(args.pretrain_dir, model_name)
        print('pretrain_model:', model_name)
        args.results_file_name = args.dataset + '_' + str(args.n_ind_class) + '_divide_seed_' + str(args.divide_seed) +'_new'
        manager = DiscoveryModelManager(pretrain_model_path, args)
        for i in range(0, args.max_stage):
            args.stage = i + 1
            wandb.init(project=args.wandb_project_name,
                       name='stage_' + str(args.stage),
                       mode=args.wandb_mode)
            data.setup(stage=args.stage)
            args.n_train_example = data.get_n_train_example(args.stage)
            manager.update_manager(args)  # Update information related to manager and stage
            manager.train(args, data.train_dataloader(), memory, data.val_dataloader(), data.test_dataloader())
            manager.test(data.test_dataloader())
            manager.save_results(args, memory.size)
            if args.use_memory:
                memory.select_ood_exemplars_to_store(manager.best_p_train_dataloader, manager.model,
                                                     manager.n_new_class)  #  Storing OOD exemplars by using pseudo-labels
                memory.print_memory_info()
        print('Training finished!')


if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    parser = ArgumentParser()
    # setting
    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")
    parser.add_argument('--pretrain', action="store_true")
    parser.add_argument('--train', action="store_true")
    parser.add_argument("--freeze_bert_parameters", action="store_true")
    parser.add_argument("--gpu_id", type=str, default='0')

    # dataset
    parser.add_argument("--dataset", default="banking", type=str, help="dataset")
    parser.add_argument("--n_ind_class", default=32, type=int, help="number of ind classes")
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--bert_model", default="bert-base-uncased",
                        type=str,
                        help="backbone architecture")
    parser.add_argument('--divide_seed', type=int, default=0, help="seed for dataset partitioning only",choices=[0,10,20])

    # model
    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")
    parser.add_argument("--n_head", default=1, type=int, help="number of heads for clustering")
    parser.add_argument("--n_view", default=2, type=int, help="number of views of a sample")

    # pretrain
    parser.add_argument("--pretrain_dir", default='your_project_path/pretrain_models', type=str)
    parser.add_argument("--lr_pre", default=5e-5, type=float)
    parser.add_argument("--pretrain_epoch", default=100, type=int)
    parser.add_argument("--pretrain_batch_size", default=64, type=int)

    # train
    parser.add_argument("--train_epoch", default=30, type=int, help="The training epochs.")
    parser.add_argument("--lr_train", default=0.4, type=float)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--instance_temperature", default=0.5, type=float,
                        help="instance contrastive learning temperature")
    parser.add_argument('--proto_m', default=0.99, type=float,
                        help='momentum for computing the momving average of prototypes')
    
    # pretrain & train
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--wait_patient", default=10, type=int)

    # Memory
    parser.add_argument('--use_memory', action="store_true")
    parser.add_argument("--n_exemplar_per_class", default=5, type=int, help='5 or 10 or 1000(denote all)')

    # log & output
    parser.add_argument("--wandb_mode", default='offline', type=str, help="offline or online")
    parser.add_argument("--save_results_path", default='results', type=str)
    parser.add_argument("--wandb_project_name", default='Baseline', type=str)
    # others
    parser.add_argument("--log_dir", default="your_project_path/logs", type=str)

    args = parser.parse_args()
    main(args)
