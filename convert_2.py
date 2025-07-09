import rasterio

input_img_file = "input.img"
output_tif_file = "output.tiff"

# Open the .img file with Rasterio
with rasterio.open(input_img_file) as src:
    # Read the image data as a NumPy array
    img_array = src.read()

    # Get the metadata from the source file
    profile = src.profile

    # Update the driver to GeoTIFF and specify the output file name
    profile.update(
        driver='GTiff',
        dtype=img_array.dtype,
        count=src.count,
        compress='lzw'  # Optional: add compression
    )

    # Write the array to a new GeoTIFF file
    with rasterio.open(output_tif_file, 'w', **profile) as dst:
        dst.write(img_array)

print(f"Successfully converted {input_img_file} to {output_tif_file}")
