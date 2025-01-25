import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import Model, MainDataset, train_model, evaluate_model
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
    output_dim = args.output_dim
    num_epochs = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train_num, x_train_cat, x_val_num, x_val_cat, x_test_num, x_test_cat, y_train, y_val, y_test, num_feature_info, cat_feature_info = split_df(year=2024, month=1)

    # x_train_cat_scaled = []
    # x_val_cat_scaled = []
    # x_test_cat_scaled = []

    # for train_cat_feat, val_cat_feat, test_cat_feat in zip(x_train_cat, x_val_cat, x_test_cat):
    #     x_train_cat_scaled.append(torch.tensor(train_cat_feat[:10], dtype=torch.int))
    #     x_val_cat_scaled.append(torch.tensor(val_cat_feat[:10], dtype=torch.int))
    #     x_test_cat_scaled.append(torch.tensor(test_cat_feat[:10], dtype=torch.int)) 
    
    # x_train_cat = x_train_cat_scaled
    # x_val_cat = x_val_cat_scaled
    # x_test_cat = x_test_cat_scaled

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
    args = parser.parse_args()
    main(args)


# code for git push
# git add .
# git commit -m "pretrain"
# git branch -m main
# git push -u origin main
