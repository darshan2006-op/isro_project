import os
import subprocess
from lxml import etree

# ---- Config ----
IMG_FILE = "C:/Users/darsh/Downloads/ch2_ohr_ncp_20250612T2031048828_d_img_d18/data/calibrated/20250612/ch2_ohr_ncp_20250612T2031048828_d_img_d18.img"
XML_FILE = "C:/Users/darsh/Downloads/ch2_ohr_ncp_20250612T2031048828_d_img_d18/data/calibrated/20250612/ch2_ohr_ncp_20250612T2031048828_d_img_d18.xml"  # or .html
TIFF_FILE = "output.tiff"

# ---- Step 1: Parse Metadata from XML ----
def parse_xml(xml_path):
    with open(xml_path, 'rb') as f:
        tree = etree.parse(f)
    ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}

    samples = int(tree.find('.//pds:Axis_Array[pds:axis_name="Sample"]/pds:elements', ns).text)
    lines = int(tree.find('.//pds:Axis_Array[pds:axis_name="Line"]/pds:elements', ns).text)
    data_type = tree.find('.//pds:Element_Array/pds:data_type', ns).text

    # Map PDS data_type to ENVI
    dt_map = {
        "UnsignedByte": 1,
        "SignedByte": 1,
        "UnsignedInteger": 12,
        "SignedInteger": 2,
        "IEEE754MSBSingle": 4,
    }
    envi_dtype = dt_map.get(data_type, 1)

    return samples, lines, envi_dtype

# ---- Step 2: Generate HDR file ----
def create_hdr(img_path, samples, lines, data_type):
    hdr_path = img_path.replace(".img", ".hdr")
    hdr_content = f"""ENVI
samples = {samples}
lines   = {lines}
bands   = 1
header offset = 0
file type = ENVI Standard
data type = {data_type}
interleave = bsq
byte order = 0
"""
    with open(hdr_path, 'w') as f:
        f.write(hdr_content)
    return hdr_path

# ---- Step 3: Run GDAL Translate ----
def convert_to_tiff(img_path, tiff_path):
    cmd = [
        "gdal_translate",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        img_path,
        tiff_path
    ]
    subprocess.run(cmd, check=True)

# ---- Main Flow ----
def main():
    samples, lines, data_type = parse_xml(XML_FILE)
    print(f"Parsed XML: {samples=} {lines=} {data_type=}")
    create_hdr(IMG_FILE, samples, lines, data_type)
    print(f"Generated .hdr file. Now converting to TIFF...")
    convert_to_tiff(IMG_FILE, TIFF_FILE)
    print(f"Done! TIFF saved at: {TIFF_FILE}")

if __name__ == "__main__":
    main()

