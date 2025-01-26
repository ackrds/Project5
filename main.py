import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import Model, MainDataset, train_model, evaluate_model
from pretrain import PretrainingModel, PretrainingDataset, pretrain_model
from loss_utils import HybridLoss
from preprocessing import hash_features, split_df
from mambular.base_models import Mambular, FTTransformer, SAINT, MambAttention
from mambular.configs import DefaultFTTransformerConfig


def main(args):

    # training parameters
    batch_size = args.batch_size
    pretrain_epochs = args.pretrain_epochs
    learning_rate = args.learning_rate
    model_to_use = args.model_type
    pretrain = args.pretrain
    num_epochs = args.num_epochs

    # custom output parameters
    output_dim = args.output_dim

    # model parameters
    d_model = args.d_model # embedding dimension
    transformer_dim_feedforward = args.transformer_dim_feedforward
    n_layers = args.n_layers
    n_heads = args.n_heads
    ff_dropout = args.ff_dropout 
    pooling_method = args.pooling_method
    use_cls = args.use_cls
    embedding_type = args.embedding_type


    # data parameters
    year = args.year
    month = args.month


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_num, x_train_cat, x_val_num, x_val_cat, x_test_num, x_test_cat, y_train, y_val, y_test, num_feature_info, cat_feature_info = split_df(year=year, month=month)

    x_train_cat_scaled = []
    x_val_cat_scaled = []
    x_test_cat_scaled = []

    for train_cat_feat, val_cat_feat, test_cat_feat in zip(x_train_cat, x_val_cat, x_test_cat):
        x_train_cat_scaled.append(torch.tensor(train_cat_feat[:10], dtype=torch.int))
        x_val_cat_scaled.append(torch.tensor(val_cat_feat[:10], dtype=torch.int))
        x_test_cat_scaled.append(torch.tensor(test_cat_feat[:10], dtype=torch.int)) 
    
    x_train_cat = x_train_cat_scaled
    x_val_cat = x_val_cat_scaled
    x_test_cat = x_test_cat_scaled

    y_train = y_train[:10]
    y_val = y_val[:10]
    y_test = y_test[:10]

    scaler = StandardScaler()
    x_train_num_scaled = []
    x_val_num_scaled = []
    x_test_num_scaled = []
    for train_feat, val_feat, test_feat in zip(x_train_num, x_val_num, x_test_num):
        x_train_num_scaled.append(torch.tensor(scaler.fit_transform(train_feat[:10]), dtype=torch.float32))
        x_val_num_scaled.append(torch.tensor(scaler.transform(val_feat[:10]), dtype=torch.float32))
        x_test_num_scaled.append(torch.tensor(scaler.transform(test_feat[:10]), dtype=torch.float32))

    # scaler = StandardScaler()
    # x_train_num_scaled = []
    # x_val_num_scaled = []
    # x_test_num_scaled = []
    # for train_feat, val_feat, test_feat in zip(x_train_num, x_val_num, x_test_num):
    #     x_train_num_scaled.append(torch.tensor(scaler.fit_transform(train_feat), dtype=torch.float32))
    #     x_val_num_scaled.append(torch.tensor(scaler.transform(val_feat), dtype=torch.float32))
    #     x_test_num_scaled.append(torch.tensor(scaler.transform(test_feat), dtype=torch.float32))

    
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

    config = DefaultFTTransformerConfig()
    config.d_model = d_model
    config.n_layers = n_layers
    config.n_heads = n_heads
    config.ff_dropout = ff_dropout
    config.transformer_dim_feedforward = transformer_dim_feedforward
    config.pooling_method = pooling_method
    config.use_cls = use_cls
    config.embedding_type = embedding_type  


    model = Model(
        cat_feature_info=cat_feature_info,
        num_feature_info=num_feature_info,
        model=model_to_use,
        output_dim=output_dim,
        pretrained_state_dict=pretrained_state_dict
    ).to(device)
    #
    # # print(model.state_dict.keys())
    #
    train_dataset = MainDataset(x_train_num_scaled, x_train_cat, y_train)
    val_dataset  = MainDataset(x_val_num_scaled, x_val_cat, y_val)
    test_dataset = MainDataset(x_test_num_scaled, x_test_cat, y_test)
    #
    # # Create training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    criterion = torch.nn.CrossEntropyLoss()
    # criterion =  HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

    # Test the model
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loss, test_accuracy = evaluate_model(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Finished testing period {year}-{month}")         
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("main", add_help=True)
    parser.add_argument("--pretrain_epochs", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--model_type", type=lambda x: eval(x), default='FTTransformer', help="type of model to use")
    parser.add_argument("--pretrain", type=int, default=0, help="pretrain the model")
    parser.add_argument("--output_dim", type=int, default=32, help="output dimension")
    parser.add_argument("--d_model", type=int, default=128, help="d_model")
    parser.add_argument("--transformer_dim_feedforward", type=int, default=256, help="transformer dim_feedforward")
    parser.add_argument("--n_layers", type=int, default=4, help="n_layers")
    parser.add_argument("--n_heads", type=int, default=8, help="n_heads")
    parser.add_argument("--ff_dropout", type=float, default=0.1, help="ff_dropout")
    parser.add_argument("--use_cls", type=bool, default=False, help="use cls")
    parser.add_argument("--pooling_method", type=str, default='avg', help="pooling method")
    parser.add_argument("--embedding_type", type=str, default='linear', help="embedding type")
    parser.add_argument("--year", type=int, default=2024, help="year")
    parser.add_argument("--month", type=int, default=1, help="month")

    args = parser.parse_args()
    main(args)


# code for git push
# git add .
# git commit -m "pretrain"
# git branch -m main
# git push -u origin main
