# The following pseudocode illustrates the loading of the configuration file.
import configparser

# Load configuration from a file
def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config
# Example usage
config_path = 'sample_pipeline_config.ini'
config = load_config(config_path)