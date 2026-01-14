import yaml

def get_config():
    with open('rby1-data-collection/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config

config = get_config()

root_path = config['demo_root'] + config['task_name']

print(root_path)