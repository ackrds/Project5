import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from model import Model, MainDataset, train_model, evaluate_model
from model import Model, MainDataset, train_models, evaluate_models
from pretrain import PretrainingModel, PretrainingDataset, pretrain_model
from loss_utils import HybridLoss
from preprocessing import hash_features, split_df
from mambular.base_models import Mambular, FTTransformer, SAINT, MambAttention
from mambular.configs import DefaultFTTransformerConfig
from utils import calculate_multipliers


def main(args):

    # training parameters
    batch_size = args.batch_size
    pretrain_epochs = args.pretrain_epochs
    learning_rate = args.learning_rate
    model_to_use = args.model_type
    pretrain = args.pretrain
    num_epochs = args.num_epochs
    sce_weight = args.sce_weight
    ce_weight = args.ce_weight
    gamma = args.gamma

    # custom output parameters
    output_dim = args.output_dim

    # model parameters
    d_model = args.d_model # embedding dimension
    transformer_dim_feedforward = args.transformer_dim_feedforward
    n_layers = args.n_layers
    n_heads = args.n_heads
    pooling_method = args.pooling_method
    use_cls = args.use_cls
    embedding_type = args.embedding_type
    attn_dropout = args.attn_dropout

    # data parameters
    year = args.year
    month = args.month


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_num, x_train_cat, x_val_num, x_val_cat, x_test_num, x_test_cat, y_train, y_val, y_test, num_feature_info, cat_feature_info, test_columns = split_df(year=year, month=month)

    # x_train_cat_scaled = []
    # x_val_cat_scaled = []
    # x_test_cat_scaled = []

    # for train_cat_feat, val_cat_feat, test_cat_feat in zip(x_train_cat, x_val_cat, x_test_cat):
    #     x_train_cat_scaled.append(torch.tensor(train_cat_feat[:10], dtype=torch.int))
    #     x_val_cat_scaled.append(torch.tensor(val_cat_feat[:10], dtype=torch.int))
    #     x_test_cat_scaled.append(torch.tensor(test_cat_feat[:1000], dtype=torch.int)) 
    
    # x_train_cat = x_train_cat_scaled
    # x_val_cat = x_val_cat_scaled
    # x_test_cat = x_test_cat_scaled

    # y_train = y_train[:10]
    # y_val = y_val[:10]
    # y_test = y_test[:1000]

    # scaler = StandardScaler()
    # x_train_num_scaled = []
    # x_val_num_scaled = []
    # x_test_num_scaled = []
    # for train_feat, val_feat, test_feat in zip(x_train_num, x_val_num, x_test_num):
    #     x_train_num_scaled.append(torch.tensor(scaler.fit_transform(train_feat[:10]), dtype=torch.float32))
    #     x_val_num_scaled.append(torch.tensor(scaler.transform(val_feat[:10]), dtype=torch.float32))
    #     x_test_num_scaled.append(torch.tensor(scaler.transform(test_feat[:1000]), dtype=torch.float32))

    scaler = StandardScaler()
    x_train_num_scaled = []
    x_val_num_scaled = []
    x_test_num_scaled = []
    for train_feat, val_feat, test_feat in zip(x_train_num, x_val_num, x_test_num):
        x_train_num_scaled.append(torch.tensor(scaler.fit_transform(train_feat), dtype=torch.float32))
        x_val_num_scaled.append(torch.tensor(scaler.transform(val_feat), dtype=torch.float32))
        x_test_num_scaled.append(torch.tensor(scaler.transform(test_feat), dtype=torch.float32))

    
    config = DefaultFTTransformerConfig()
    config.d_model = d_model
    config.n_layers = n_layers
    config.n_heads = n_heads
    config.transformer_dim_feedforward = transformer_dim_feedforward
    config.pooling_method = pooling_method
    config.use_cls = use_cls
    config.embedding_type = embedding_type  
    config.attn_dropout = attn_dropout


    if pretrain==1:
        pretrain_dataset = PretrainingDataset(
            x_train_num_scaled, x_train_cat, cat_feature_info, mask_ratio=0.25
        )

        preval_dataset = PretrainingDataset(
            x_val_num_scaled, x_val_cat, cat_feature_info, mask_ratio=0.25
        )
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        preval_loader = DataLoader(
                preval_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
        
        # Initialize and pretrain model
        pretrain_model_inst = PretrainingModel(
            cat_feature_info,
            num_feature_info,
            model_to_use,
            output_dim,
            config=config
        ).to(device)

        # Pretrain the model
        pretrained_model = pretrain_model(
            pretrain_model_inst,
            pretrain_loader,
            pretrain_loader,
            pretrain_epochs,
            device
        )

        pretrained_state_dict = pretrained_model.encoder.state_dict()

    else:
        pretrained_state_dict = None


    model1 = Model(
        cat_feature_info=cat_feature_info,
        num_feature_info=num_feature_info,
        model=model_to_use,
        output_dim=output_dim,
        pretrained_state_dict=pretrained_state_dict,
        config=config
    ).to(device)

    model2 = Model(
        cat_feature_info=cat_feature_info,
        num_feature_info=num_feature_info,
        model=model_to_use,
        output_dim=output_dim,
        pretrained_state_dict=pretrained_state_dict,
        config=config
    ).to(device)

    model3 = Model(
        cat_feature_info=cat_feature_info,
        num_feature_info=num_feature_info,
        model=model_to_use,
        output_dim=output_dim,
        pretrained_state_dict=pretrained_state_dict,
        config=config
    ).to(device)

    train_dataset = MainDataset(x_train_num_scaled, x_train_cat, y_train)
    val_dataset  = MainDataset(x_val_num_scaled, x_val_cat, y_val)
    test_dataset = MainDataset(x_test_num_scaled, x_test_cat, y_test)

    train_loader1 = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    val_loader1 = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    train_loader2 = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    val_loader2 = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    train_loader3 = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    val_loader3 = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # criterion = torch.nn.CrossEntropyLoss()
    criterion =  HybridLoss(ce_weight=ce_weight, sce_weight=sce_weight)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=gamma)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=gamma)

    optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=5, gamma=gamma)

    models = [model1, model2, model3]
    train_loaders = [train_loader1, train_loader2, train_loader3]
    val_loaders = [val_loader1, val_loader2, val_loader3]
    criterions = [criterion, criterion, criterion]
    optimizers = [optimizer1, optimizer2, optimizer3]
    schedulers = [scheduler1, scheduler2, scheduler3]

    models, histories  = train_models(models, train_loaders, val_loaders, criterions, optimizers, num_epochs, device, schedulers)
    
    # trained_model, history = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     num_epochs=num_epochs,
    #     device=device,
    #     scheduler=scheduler
    # )

    # Test the model
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )

    # test_loss, test_accuracy, test_preds = evaluate_model(
    #     model=trained_model,
    #     data_loader=test_loader,
    #     criterion=criterion,
    #     device=device
    # )

    # print(f"Finished testing period {year}-{month}")         
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    # calculate_multipliers(test_preds.cpu().numpy(), test_columns)


if __name__ == "__main__":
    seed = 42  # You can choose any integer as the seed
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)


    parser = argparse.ArgumentParser("main", add_help=True)
    parser.add_argument("--pretrain_epochs", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--model_type", type=lambda x: eval(x), default='FTTransformer', help="type of model to use")
    parser.add_argument("--pretrain", type=int, default=0, help="pretrain the model")
    parser.add_argument("--sce_weight", type=float, default=0.1, help="sce weight")
    parser.add_argument("--ce_weight", type=float, default=0.1, help="ce weight")
    parser.add_argument("--output_dim", type=int, default=32, help="output dimension")
    parser.add_argument("--d_model", type=int, default=128, help="d_model")
    parser.add_argument("--transformer_dim_feedforward", type=int, default=256, help="transformer dim_feedforward")
    parser.add_argument("--n_layers", type=int, default=4, help="n_layers")
    parser.add_argument("--n_heads", type=int, default=8, help="n_heads")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="attn_dropout")
    parser.add_argument("--use_cls", type=bool, default=False, help="use cls")
    parser.add_argument("--pooling_method", type=str, default='avg', help="pooling method")
    parser.add_argument("--embedding_type", type=str, default='linear', help="embedding type")
    parser.add_argument("--year", type=int, default=2024, help="year")
    parser.add_argument("--month", type=int, default=1, help="month")
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma")
    args = parser.parse_args()
    main(args)


# code for git push
# git add .
# git commit -m "pretrain"
# git branch -m main
# git push -u origin main
