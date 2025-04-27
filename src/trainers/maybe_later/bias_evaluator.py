class BiasEvaluation(ABC):
    def __init__(
        self,
        dataset_name: DatasetName,
        config: dict,
        checkpoint_path: Union[Path, str],
        wandb_project_name: str = "PASSION-Evaluation",
        log_wandb: bool = False,
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.wandb_project_name = wandb_project_name
        self.log_wandb = log_wandb
        self.seed = config["seed"]
        fix_random_seeds(self.seed)

        # Setup paths
        self.output_path = Path(config.get("output_path", "assets/evaluation"))
        self.cache_path = Path(config.get("cache_path", "assets/evaluation/cache"))

        self.df_name = f"{self.experiment_name}_{self.dataset_name.value}.csv"
        self.df_path = self.output_path / self.df_name
        self.model_path = self.output_path / self.experiment_name

        self.df = pd.DataFrame(
            [],
            columns=[
                "Score",
                "EvalTargets",
                "EvalPredictions",
                "EvalType",
                "AdditionalRunInfo",
                "SplitName",
            ],
        )

        # Ensure paths exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Load the dataset and model
        self.dataset, self.torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(config["dataset"][dataset_name.value]["path"]),
            batch_size=config.get("batch_size", 128),
            transform=self.create_transform(),
            **config["dataset"][dataset_name.value],
        )
        self.model, self.model_out_dim = self.load_model_from_checkpoint()

    def create_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def load_model_from_checkpoint(self):
        model, info, _ = Embedder.load_pretrained(
            "imagenet", return_info=True, n_head_layers=0
        )
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model, info.out_dim

    def evaluate(self):
        for e_type, config in self.config.items():
            if e_type in eval_type_dict:
                for (
                    train_valid_range,
                    test_range,
                    split_name,
                ) in self.split_dataframe_iterator():
                    # Do evaluation logic here
                    self._run_evaluation_on_range(
                        e_type, train_valid_range, test_range, split_name
                    )

    def _run_evaluation_on_range(
        self,
        e_type: BaseEvalType,
        train_range: np.ndarray,
        eval_range: np.ndarray,
        split_name: Optional[str] = None,
    ):
        # Perform evaluation here, using the loaded model
        score_dict = e_type.evaluate(
            self.torch_dataset, self.model, train_range, eval_range
        )
        self.save_results(score_dict, split_name)

    def save_results(self, score_dict, split_name):
        # Save the evaluation results
        self.df.loc[len(self.df)] = [
            score_dict["score"],
            score_dict["targets"],
            score_dict["predictions"],
            e_type.name(),
            split_name,
        ]
        self.df.to_csv(self.df_path, index=False)

    @property
    @abstractmethod
    def experiment_name(self):
        pass

    @abstractmethod
    def split_dataframe_iterator(self):
        pass
