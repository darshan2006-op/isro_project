import xml.etree.ElementTree as ET
import json

# Define your input XML file
input_file = "C:/Users/darsh/Downloads/ch2_ohr_ncp_20240930T2025441752_d_img_d18/data/calibrated/20240930/input.xml"  # or .html

# Parse the XML file with namespace awareness
tree = ET.parse(input_file)
root = tree.getroot()

# Extract all namespace mappings
namespaces = dict([
    node for _, node in ET.iterparse(input_file, events=['start-ns'])
])

# Utility function to strip namespace from tag
def strip_tag(tag):
    return tag.split('}', 1)[-1] if '}' in tag else tag

# Recursively convert XML to dictionary
def xml_to_dict(element):
    tag = strip_tag(element.tag)
    data = {}

    # Attributes
    if element.attrib:
        data.update({f"@{k}": v for k, v in element.attrib.items()})

    # Children
    children = list(element)
    if children:
        child_dict = {}
        for child in children:
            child_tag = strip_tag(child.tag)
            child_data = xml_to_dict(child)
            if child_tag in child_dict:
                if not isinstance(child_dict[child_tag], list):
                    child_dict[child_tag] = [child_dict[child_tag]]
                child_dict[child_tag].append(child_data)
            else:
                child_dict[child_tag] = child_data
        data.update(child_dict)
    else:
        # Leaf node
        text = element.text.strip() if element.text else ""
        data = text if not data else {**data, "#text": text}

    return data

# Convert root to dictionary
json_data = {strip_tag(root.tag): xml_to_dict(root)}

# Output JSON
output_file = "output.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4)

print(f"Converted XML to JSON and saved to {output_file}")
