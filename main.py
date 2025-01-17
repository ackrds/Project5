import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import Model, MainDataset, train_model
from pretrain import PretrainingModel, PretrainingDataset, pretrain_model
from loss_utils import HybridLoss
from preprocessing import hash_features, split_df
from mambular.base_models import Mambular, FTTransformer, SAINT, MambAttention


def main(args):

    batch_size = args.batch_size
    pretrain_epochs = args.pretrain_epochs
    learning_rate = args.learning_rate
    model_to_use = args.model_type
    pretrain = args.pretrain
    print(learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_num, x_train_cat, x_val_num, x_val_cat, y_train, y_val, num_feature_info, cat_feature_info = split_df()

    print(num_feature_info)
    scaler = StandardScaler()
    x_train_num_scaled = []
    x_val_num_scaled = []
    for train_feat, val_feat in zip(x_train_num, x_val_num):
        x_train_num_scaled.append(torch.tensor(scaler.fit_transform(train_feat), dtype=torch.float32))
        x_val_num_scaled.append(torch.tensor(scaler.transform(val_feat), dtype=torch.float32))
    
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
            model_to_use
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

    model = Model(
        cat_feature_info=cat_feature_info,
        num_feature_info=num_feature_info,
        model=model_to_use,
        pretrained_state_dict=pretrained_state_dict
    ).to(device)
    #
    # # print(model.state_dict.keys())
    #
    train_dataset = MainDataset(x_train_num_scaled, x_train_cat, y_train)
    val_dataset  = MainDataset(x_val_num_scaled, x_val_cat, y_val)
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

    num_epochs = args.num_epochs
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
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("main", add_help=True)
    parser.add_argument("--pretrain_epochs", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--model_type", type=lambda x: eval(x), default='FTTransformer', help="type of model to use")
    parser.add_argument("--pretrain", type=int, default=0, help="pretrain the model")
    args = parser.parse_args()
    main(args)


# code for git push
# git add .
# git commit -m "pretrain"
# git branch -m main
# git push -u origin main