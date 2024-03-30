from sagemaker.model_monitor import DefaultModelMonitor

baseline_data_uri = 's3://your-s3-bucket/baseline_data/'
my_default_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600
)

my_default_monitor.suggest_baseline(
    baseline_dataset=baseline_data_uri,
    dataset_format=DefaultModelMonitor.DATASET_FORMAT.csv(header=True),
)
