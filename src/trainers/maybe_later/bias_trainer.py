class Trainer(ABC):
    def __init__(
        self,
        dataset_name: DatasetName,
        config: dict,
        wandb_project_name: str = "PASSION-Training",
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.wandb_project_name = wandb_project_name
        self.seed = config["seed"]
        fix_random_seeds(self.seed)

        # Setup paths
        self.output_path = Path(config.get("output_path", "assets/training"))
        self.cache_path = Path(config.get("cache_path", "assets/training/cache"))

        self.model_path = self.output_path / f"{self.experiment_name}_model.pth"

        # Load the dataset and model
        self.dataset, self.torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(config["dataset"][dataset_name.value]["path"]),
            batch_size=config.get("batch_size", 128),
            transform=self.create_transform(),
            **config["dataset"][dataset_name.value],
        )
        self.model, self.model_out_dim = self.load_model_for_training()

    def create_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def load_model_for_training(self):
        # Define and initialize model
        model, info, _ = Embedder.load_pretrained(
            "imagenet", return_info=True, n_head_layers=0
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model, info.out_dim

    def train(self):
        # Implement the training loop here
        # Save the model after training
        torch.save(self.model.state_dict(), self.model_path)

    @property
    @abstractmethod
    def experiment_name(self):
        pass
