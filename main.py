import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model import Model, MainDataset, train_model, evaluate_model
from pretrain import PretrainingModel, PretrainingDataset, pretrain_model
from loss_utils import HybridLoss, FocalLoss
from preprocessing import hash_features, split_df
from saint.model import SAINT
from utils import calculate_multipliers
from mambular.base_models import  * 
from mambular.configs import *


# from models.config import DefaultFTTransformerConfig as DefaultCustomFTTransformerConfig

# Constants for magic numbers
SAMPLE_SIZE = 10
LARGE_SAMPLE_SIZE = 1000

def scale_features(scaler_type, x_train_num, x_val_num, x_test_num):
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    x_train_num_scaled = []
    x_val_num_scaled = []
    x_test_num_scaled = []
    for train_feat, val_feat, test_feat in zip(x_train_num, x_val_num, x_test_num):
        x_train_num_scaled.append(torch.tensor(scaler.fit_transform(train_feat), dtype=torch.float32))
        x_val_num_scaled.append(torch.tensor(scaler.transform(val_feat), dtype=torch.float32))
        x_test_num_scaled.append(torch.tensor(scaler.transform(test_feat), dtype=torch.float32))
    return x_train_num_scaled, x_val_num_scaled, x_test_num_scaled

def load_datasets(x_train_num_scaled, x_train_cat, y_train, x_val_num_scaled, x_val_cat, y_val, x_test_num_scaled, x_test_cat, y_test):
    train_dataset = MainDataset(x_train_num_scaled, x_train_cat, y_train)
    val_dataset = MainDataset(x_val_num_scaled, x_val_cat, y_val)
    test_dataset = MainDataset(x_test_num_scaled, x_test_cat, y_test)
    return train_dataset, val_dataset, test_dataset

def main(args):

    # DefaultCustomFTTransformerConfig = {}
    
    # training parameters
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    pretrain = args.pretrain
    n_bins = args.n_bins
    pretrain_config_dict = args.pretrain_config
    pretrain_epochs = args.pretrain_epochs
    pretrain_learning_rate = args.pretrain_learning_rate
    learning_rate = args.learning_rate
    model_to_use = args.model_type
    num_epochs = args.num_epochs
    use_embeddings = args.use_embeddings
    sce_weight = args.sce_weight
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    ce_weight = args.ce_weight
    t_max = args.t_max
    eta_min = args.eta_min
    weight_decay = args.weight_decay
    verbose = args.verbose
    patience = args.patience
    l2_lambda = args.l2_lambda
    output_dim = args.output_dim
    criterion = args.criterion
    # data parameters
    year = args.year
    month = args.month
    scaler = args.scaler


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_num, x_train_cat, x_val_num, x_val_cat, x_test_num, x_test_cat, y_train, y_val, y_test, num_feature_info, cat_feature_info, test_columns = split_df(year=year, month=month)
    x_train_num_scaled, x_val_num_scaled, x_test_num_scaled = scale_features(scaler, x_train_num, x_val_num, x_test_num)

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

    print(len(x_train_num_scaled[0]))
    if use_embeddings == 1:
        x_train_cat = [f.unsqueeze(1) for f in x_train_cat]
        x_val_cat = [f.unsqueeze(1) for f in x_val_cat]
        x_test_cat = [f.unsqueeze(1) for f in x_test_cat]

    if model_to_use == "DenseFTTransformer" or model_to_use == "Fastformer":
        config = DefaultFTTransformerConfig()
    else:
        config = eval(f"Default{model_to_use}Config()")
    model_to_use = eval(f"{model_to_use}")


    if len(args.config_values.keys()) > 0:
        for key, value in args.config_values.items():
            setattr(config, key, value)


    if pretrain==1:

        pretrain_dataset = PretrainingDataset(
            x_train_num_scaled, x_train_cat, cat_feature_info
        )

        preval_dataset = PretrainingDataset(
            x_val_num_scaled, x_val_cat, cat_feature_info
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
        pretrain_config = {}
        for key, value in pretrain_config_dict.items():
            pretrain_config[key] = value
        
        # Initialize and pretrain model
        pretrain_model_inst = PretrainingModel(
            cat_feature_info,
            num_feature_info,
            model_to_use,
            output_dim,
            pretrain_config=pretrain_config,
            config=config
        ).to(device)

        # print(pretrain_model_inst.encoder.embedding_layer.state_dict)

        pretrained_model = pretrain_model(
            pretrain_model_inst,
            pretrain_loader,
            preval_loader,
            pretrain_epochs,
            device, 
            lr=pretrain_learning_rate,
        )

        # pretrained_state_dict = pretrained_model.encoder.embedding_layer.state_dict()

    else:
        pretrained_model = None

    model = Model(
            cat_feature_info=cat_feature_info,
            num_feature_info=num_feature_info,
            model=model_to_use,
            output_dim=output_dim,
            config=config,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)
    
    if pretrained_model is not None:
        print("Loading pretrained model")
        model.model.embedding_layer.load_state_dict(pretrained_model.embedding_layer.state_dict())
        model.model.encoder.load_state_dict(pretrained_model.encoder.state_dict())


    train_dataset, val_dataset, test_dataset = load_datasets(x_train_num_scaled, x_train_cat, y_train, x_val_num_scaled, x_val_cat, y_val, x_test_num_scaled, x_test_cat, y_test)

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

    # criterion = torch.nn.CrossEntropyLoss()
    if criterion == "focal":
        criterion =  FocalLoss(alpha=1, gamma=2)
    else:
        criterion = HybridLoss(n_bins=n_bins, ce_weight=ce_weight, sce_weight=sce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=t_max,  # Total number of epochs
    eta_min=eta_min  # Minimum learning rate
    ) 

    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        scheduler=scheduler,
        verbose=verbose,
        patience=patience,
        l2_lambda=l2_lambda
    )

    # Test the model
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loss, test_accuracy, test_preds = evaluate_model(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Finished testing period {year}-{month}")         
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    calculate_multipliers(test_preds.cpu().numpy(), test_columns)

if __name__ == "__main__":
    seed = 42  # You can choose any integer as the seed
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    def parse_dict(string):
        try:
            return json.loads(string)
        except json.JSONDecodeError:
            raise argparse.ArgumentTypeError("Invalid dictionary format")


    parser = argparse.ArgumentParser("main", add_help=True)
    
    # Data
    parser.add_argument("--scaler", type=str, default='standard', help="scaler to use")
    parser.add_argument("--year", type=int, default=2024, help="year")
    parser.add_argument("--month", type=int, default=1, help="month")

    # Pretraining
    parser.add_argument("--pretrain", type=int, default=0, help="pretrain the model")
    parser.add_argument("--pretrain_epochs", type=int, default=15, help="epochs")
    parser.add_argument("--pretrain_learning_rate", type=float, default=0.001, help="pretrain learning rate")
    parser.add_argument("--pretrain_batch_size", type=int, default=512, help="pretrain batch size")
    parser.add_argument("--pretrain_config", type=parse_dict, default='{"temperature": 0.07, "dropout": 0.2, "hidden_dim": 1024, "projection_dim": 512, "lambda_": 0.75, "numeric_loss_type": "mae"}', help="pretrain config")

    # Training
    parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")

    # Model
    parser.add_argument("--model_type", type=str, default='FTTransformer', help="type of model to use")
    parser.add_argument("--output_dim", type=int, default=256, help="output dimension")
    parser.add_argument("--use_embeddings", type=int, default=0, help="use embeddings")
    parser.add_argument("--config_values", type=parse_dict, default='{"d_model": 256, "transformer_dim_feedforward": 1024, "output_dim":256, "ff_dropout":0.2, "attn_dropout":0.2}', help="config_dict")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--hidden_dim", type=int, default=512, help="hidden dimension")
    parser.add_argument("--patience", type=int, default=100, help="patience")
    # Test
    parser.add_argument("--test_batch_size", type=int, default=512, help="test batch size")

    # Loss
    parser.add_argument("--criterion", type=str, default="hybrid", help="criterion")
    parser.add_argument("--sce_weight", type=float, default=1, help="sce weight")
    parser.add_argument("--ce_weight", type=float, default=0.5, help="ce weight")
    parser.add_argument("--n_bins", type=int, default=30, help="number of bins")
    parser.add_argument("--l2_lambda", type=float, default=0.01, help="l2 lambda")
    
    # Scheduler
    parser.add_argument("--t_max", type=int, default=40, help="cosine annealing t_max")
    parser.add_argument("--eta_min", type=float, default=1e-7, help="cosine annealing eta_min")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")

    parser.add_argument("--verbose", type=int, default=0, help="verbose")

    args = parser.parse_args()
    main(args)


# code for git push
# git add .
# git commit -m "pretrain"
# git branch -m main
# git push -u origin main
  
