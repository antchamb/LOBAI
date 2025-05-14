import torch
import json
import pprint

# Get CUDA device properties
device_properties = torch.cuda.get_device_properties(0)

# Convert properties to a dictionary
device_properties_dict = {
    "name": device_properties.name,
    "total_memory": device_properties.total_memory,
    "multi_processor_count": device_properties.multi_processor_count,
    "major": device_properties.major,
    "minor": device_properties.minor,
}

# Pretty print the JSON representation
pprint.pprint(json.loads(json.dumps(device_properties_dict)))