from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri='s3://your-s3-bucket/endpoint-capture/',
    capture_options=["REQUEST", "RESPONSE"]
)
predictor = sagemaker.predictor.RealTimePredictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    data_capture_config=data_capture_config
)
