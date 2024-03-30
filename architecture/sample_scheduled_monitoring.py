from sagemaker.model_monitor import CronExpressionGenerator

my_default_monitor.create_monitoring_schedule(
    monitor_schedule_name='my-model-monitor-schedule',
    endpoint_input=predictor.endpoint,
    output_s3_uri='s3://your-s3-bucket/monitoring-output/',
    statistics=my_default_monitor.baseline_statistics(),
    constraints=my_default_monitor.suggested_constraints(),
    schedule_cron_expression=CronExpressionGenerator.daily(),
)
