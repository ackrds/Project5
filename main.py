
import torch
from torch.utils.data import Dataset, DataLoader
from mambular.base_models import Mambular, FTTransformer, SAINT, MambAttention
from sklearn.preprocessing import StandardScalera
from model import Model, MainDataset, train_model
from pretrain import PretrainingModel, PretrainingDataset, pretrain_model
from loss_utils import HybridLoss
from preprocessing import hash_features, split_df


def main():

    # Configuration
    # config = DefaultMambularConfig()
    batch_size = 1028
    pretrain_epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # FTTransformer
    # MambaAttention
    # SAINT
    # Mambular
    model_to_use = FTTransformer
    # Generate synthetic data for pretraining

    x_train_num, x_train_cat, x_val_num, x_val_cat, y_train, y_val, num_feature_info, cat_feature_info = split_df()

    print(y_train)

    scaler = StandardScaler()
    x_train_num_scaled = []
    x_val_num_scaled = []
    for train_feat, val_feat in zip(x_train_num, x_val_num):
        x_train_num_scaled.append(torch.tensor(scaler.fit_transform(train_feat), dtype=torch.float32))
        x_val_num_scaled.append(torch.tensor(scaler.transform(val_feat), dtype=torch.float32))
    
    pretrain_dataset = PretrainingDataset(
        x_train_num, x_train_cat, cat_feature_info, mask_ratio=0.25
    )
    pretrain_loader = DataLoader(
        pretrain_dataset,
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
        pretrain_epochs,
        device
    )

    pretrained_state_dict = pretrained_model.encoder.state_dict()

    model = Model(
        cat_feature_info=cat_feature_info,
        num_feature_info=num_feature_info,
        model=model_to_use,
        pretrained_state_dict=pretrained_state_dict
    ).to(device)
    #
    # # print(model.state_dict.keys())
    #
    train_dataset = MainDataset(x_train_num, x_train_cat, y_train)
    val_dataset  = MainDataset(x_val_num, x_val_cat, y_val)
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

    num_epochs = 10
    # # criterion = torch.nn.CrossEntropyLoss()
    criterion =  HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )
    #
    # # Print final training results
    # print("\nTraining completed!")
    # print(f"Final training accuracy: {history['accuracy'][-1]:.2f}%")
    # print(f"Final training loss: {history['loss'][-1]:.4f}")
    #

if __name__ == "__main__":
    main()


