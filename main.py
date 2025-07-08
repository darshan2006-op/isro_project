import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import griddata
from skimage import filters, morphology, feature, restoration
from skimage.segmentation import watershed
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rasterio
from rasterio.transform import from_bounds
import json
import os
from lxml import etree
import argparse


class LunarDEMGenerator:
    def __init__(self, image_path, metadata_path=None, output_dir="./dem_output"):
        """
        Initialize the Lunar DEM Generator

        Args:
            image_path: Path to the lunar image file
            metadata_path: Path to XML metadata file (optional)
            output_dir: Directory to save outputs
        """
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.metadata = {}
        self.image = None
        self.dem = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load image and metadata
        self.load_image()
        if metadata_path:
            self.load_metadata()

    def load_image(self):
        """Load and preprocess the lunar image"""
        print(f"Loading image: {self.image_path}")

        # Handle different image formats
        if self.image_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(self.image_path) as src:
                self.image = src.read(1).astype(np.float32)
                self.image_profile = src.profile
        else:
            self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            if self.image is None:
                raise ValueError(f"Could not load image: {self.image_path}")
            self.image = self.image.astype(np.float32)

        # Normalize image to 0-1 range
        self.image = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))
        print(f"Image loaded: {self.image.shape}")

    def load_metadata(self):
        """Load metadata from XML file"""
        if not self.metadata_path or not os.path.exists(self.metadata_path):
            print("No metadata file provided or file not found")
            return

        try:
            with open(self.metadata_path, 'rb') as f:
                tree = etree.parse(f)
            ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}

            # Extract relevant metadata for DEM generation
            metadata_elements = [
                ('sun_elevation', './/pds:solar_elevation_angle'),
                ('sun_azimuth', './/pds:solar_azimuth_angle'),
                ('spacecraft_altitude', './/pds:spacecraft_altitude'),
                ('pixel_scale', './/pds:pixel_resolution'),
                ('center_latitude', './/pds:center_latitude'),
                ('center_longitude', './/pds:center_longitude'),
                ('north_bound', './/pds:north_bounding_coordinate'),
                ('south_bound', './/pds:south_bounding_coordinate'),
                ('east_bound', './/pds:east_bounding_coordinate'),
                ('west_bound', './/pds:west_bounding_coordinate'),
            ]

            for key, xpath in metadata_elements:
                elem = tree.find(xpath, ns)
                if elem is not None:
                    try:
                        self.metadata[key] = float(elem.text)
                    except ValueError:
                        self.metadata[key] = elem.text

            print(f"Loaded metadata: {len(self.metadata)} parameters")

        except Exception as e:
            print(f"Error loading metadata: {e}")

    def preprocess_image(self):
        """Apply preprocessing to enhance lunar surface features"""
        print("Preprocessing image...")

        # Apply Gaussian blur to reduce noise
        processed = cv2.GaussianBlur(self.image, (3, 3), 0)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply((processed * 255).astype(np.uint8)).astype(np.float32) / 255.0

        # Apply unsharp masking to enhance edges
        gaussian_blur = cv2.GaussianBlur(processed, (9, 9), 2.0)
        processed = cv2.addWeighted(processed, 1.5, gaussian_blur, -0.5, 0)

        # Clip values to valid range
        processed = np.clip(processed, 0, 1)

        return processed

    def extract_terrain_features(self, processed_image):
        """Extract terrain features from the processed image"""
        print("Extracting terrain features...")

        # Calculate gradients
        grad_x = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Detect edges using Canny
        edges = feature.canny(processed_image, sigma=1.0, low_threshold=0.1, high_threshold=0.2)

        # Detect corners (potential crater rims)
        corners = feature.corner_harris(processed_image, method='eps', k=0.05, eps=1e-6)

        # Detect ridges and valleys using eigenvalues of Hessian
        hessian_xx = ndimage.gaussian_filter(processed_image, sigma=1.0, order=[2, 0])
        hessian_yy = ndimage.gaussian_filter(processed_image, sigma=1.0, order=[0, 2])
        hessian_xy = ndimage.gaussian_filter(processed_image, sigma=1.0, order=[1, 1])

        # Calculate eigenvalues
        trace = hessian_xx + hessian_yy
        det = hessian_xx * hessian_yy - hessian_xy ** 2
        lambda1 = 0.5 * (trace + np.sqrt(trace ** 2 - 4 * det + 1e-10))
        lambda2 = 0.5 * (trace - np.sqrt(trace ** 2 - 4 * det + 1e-10))

        # Ridge detection (both eigenvalues negative)
        ridges = (lambda1 < 0) & (lambda2 < 0) & (np.abs(lambda1) > np.abs(lambda2))

        # Valley detection (both eigenvalues positive)
        valleys = (lambda1 > 0) & (lambda2 > 0) & (lambda1 > lambda2)

        return {
            'gradients': (grad_x, grad_y, gradient_magnitude),
            'edges': edges,
            'corners': corners,
            'ridges': ridges,
            'valleys': valleys
        }

    def photometric_stereo_approximation(self, processed_image):
        """
        Approximate photometric stereo using sun angle information
        This is a simplified approach for single-image DEM generation
        """
        print("Applying photometric stereo approximation...")

        # Get sun elevation and azimuth from metadata
        sun_elevation = self.metadata.get('sun_elevation', 45.0)  # Default 45 degrees
        sun_azimuth = self.metadata.get('sun_azimuth', 0.0)  # Default 0 degrees

        # Convert to radians
        sun_elev_rad = np.radians(sun_elevation)
        sun_azim_rad = np.radians(sun_azimuth)

        # Calculate sun direction vector
        sun_x = np.cos(sun_elev_rad) * np.sin(sun_azim_rad)
        sun_y = np.cos(sun_elev_rad) * np.cos(sun_azim_rad)
        sun_z = np.sin(sun_elev_rad)

        # Estimate surface normals from image gradients
        grad_x = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=3)

        # Estimate surface slopes
        slope_x = np.arctan(grad_x)
        slope_y = np.arctan(grad_y)

        # Calculate surface normals
        normal_x = -np.sin(slope_x)
        normal_y = -np.sin(slope_y)
        normal_z = np.cos(np.sqrt(slope_x ** 2 + slope_y ** 2))

        # Normalize normals
        norm_magnitude = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
        normal_x /= norm_magnitude
        normal_y /= norm_magnitude
        normal_z /= norm_magnitude

        # Calculate dot product with sun direction
        dot_product = normal_x * sun_x + normal_y * sun_y + normal_z * sun_z

        # Estimate height using shape-from-shading principles
        # This is a simplified approach
        height_estimate = processed_image / (dot_product + 1e-6)

        return height_estimate

    def shape_from_shading(self, processed_image, iterations=100):
        """
        Implement shape-from-shading algorithm for height estimation
        """
        print("Applying shape-from-shading...")

        # Initialize height map
        height = np.zeros_like(processed_image)

        # Get illumination parameters
        sun_elevation = self.metadata.get('sun_elevation', 45.0)
        sun_azimuth = self.metadata.get('sun_azimuth', 0.0)

        # Convert to radians
        sun_elev_rad = np.radians(sun_elevation)
        sun_azim_rad = np.radians(sun_azimuth)

        # Light direction
        light_x = np.cos(sun_elev_rad) * np.sin(sun_azim_rad)
        light_y = np.cos(sun_elev_rad) * np.cos(sun_azim_rad)
        light_z = np.sin(sun_elev_rad)

        # Iterative shape-from-shading
        for i in range(iterations):
            # Calculate gradients
            grad_x = np.gradient(height, axis=1)
            grad_y = np.gradient(height, axis=0)

            # Calculate surface normals
            normal_x = -grad_x
            normal_y = -grad_y
            normal_z = np.ones_like(height)

            # Normalize
            norm = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
            normal_x /= norm
            normal_y /= norm
            normal_z /= norm

            # Calculate reflectance
            reflectance = normal_x * light_x + normal_y * light_y + normal_z * light_z
            reflectance = np.clip(reflectance, 0, 1)

            # Update height based on intensity difference
            intensity_diff = processed_image - reflectance
            height += 0.01 * intensity_diff  # Learning rate

            if i % 20 == 0:
                print(f"  Iteration {i}/{iterations}")

        return height

    def crater_detection_and_depth(self, processed_image):
        """
        Detect craters and estimate their depths
        """
        print("Detecting craters and estimating depths...")

        # Apply circular Hough transform to detect circular features (craters)
        # Convert to uint8 for OpenCV
        img_uint8 = (processed_image * 255).astype(np.uint8)

        # Apply median filter to reduce noise
        img_filtered = cv2.medianBlur(img_uint8, 5)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            img_filtered,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=200
        )

        crater_mask = np.zeros_like(processed_image)
        crater_depths = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"Detected {len(circles)} potential craters")

            for (x, y, r) in circles:
                # Create circular mask
                mask = np.zeros_like(processed_image)
                cv2.circle(mask, (x, y), r, 1, -1)

                # Calculate depth as difference between rim and center
                rim_mask = np.zeros_like(processed_image)
                cv2.circle(rim_mask, (x, y), r, 1, 2)
                cv2.circle(rim_mask, (x, y), max(1, r - 5), 0, -1)

                center_mask = np.zeros_like(processed_image)
                cv2.circle(center_mask, (x, y), max(1, r // 4), 1, -1)

                if np.sum(rim_mask) > 0 and np.sum(center_mask) > 0:
                    rim_intensity = np.mean(processed_image[rim_mask > 0])
                    center_intensity = np.mean(processed_image[center_mask > 0])
                    depth = rim_intensity - center_intensity
                    crater_depths.append(depth)

                    # Add to crater mask with depth
                    crater_mask[mask > 0] = depth

        return crater_mask, crater_depths

    def generate_dem(self, method='combined'):
        """
        Generate DEM using specified method

        Args:
            method: 'photometric', 'shape_from_shading', 'combined'
        """
        print(f"Generating DEM using {method} method...")

        # Preprocess image
        processed = self.preprocess_image()

        # Extract terrain features
        features = self.extract_terrain_features(processed)

        if method == 'photometric':
            # Use photometric stereo approximation
            height_map = self.photometric_stereo_approximation(processed)

        elif method == 'shape_from_shading':
            # Use shape-from-shading
            height_map = self.shape_from_shading(processed)

        elif method == 'combined':
            # Combine multiple methods
            height_ps = self.photometric_stereo_approximation(processed)
            height_sfs = self.shape_from_shading(processed, iterations=50)

            # Weighted combination
            height_map = 0.6 * height_ps + 0.4 * height_sfs

        else:
            raise ValueError(f"Unknown method: {method}")

        # Detect craters and incorporate depth information
        crater_mask, crater_depths = self.crater_detection_and_depth(processed)

        # Incorporate crater depths
        height_map = height_map - crater_mask * 0.5  # Scale crater depths

        # Apply smoothing
        height_map = ndimage.gaussian_filter(height_map, sigma=1.0)

        # Normalize to reasonable elevation range
        if 'spacecraft_altitude' in self.metadata:
            # Scale based on spacecraft altitude (rough approximation)
            altitude = self.metadata['spacecraft_altitude']
            height_scale = altitude * 0.001  # Rough scaling factor
        else:
            height_scale = 1000  # Default scale in meters

        height_map = height_map * height_scale

        # Remove outliers
        height_map = np.clip(height_map,
                             np.percentile(height_map, 5),
                             np.percentile(height_map, 95))

        self.dem = height_map

        print(f"DEM generated: {self.dem.shape}")
        print(f"Elevation range: {np.min(self.dem):.2f} to {np.max(self.dem):.2f} meters")

        return self.dem

    def save_dem(self, filename=None):
        """Save DEM as GeoTIFF"""
        if self.dem is None:
            raise ValueError("DEM not generated yet. Call generate_dem() first.")

        if filename is None:
            filename = os.path.join(self.output_dir, "lunar_dem.tif")

        # Prepare geospatial information
        if all(key in self.metadata for key in ['north_bound', 'south_bound', 'east_bound', 'west_bound']):
            transform = from_bounds(
                self.metadata['west_bound'],
                self.metadata['south_bound'],
                self.metadata['east_bound'],
                self.metadata['north_bound'],
                self.dem.shape[1],
                self.dem.shape[0]
            )
        else:
            # Default transform
            transform = from_bounds(0, 0, self.dem.shape[1], self.dem.shape[0],
                                    self.dem.shape[1], self.dem.shape[0])

        # Save as GeoTIFF
        with rasterio.open(
                filename,
                'w',
                driver='GTiff',
                height=self.dem.shape[0],
                width=self.dem.shape[1],
                count=1,
                dtype=self.dem.dtype,
                crs='EPSG:4326',  # WGS84 - adjust as needed for lunar coordinates
                transform=transform,
                compress='lzw'
        ) as dst:
            dst.write(self.dem, 1)

            # Add metadata
            dst.update_tags(
                DESCRIPTION="Lunar DEM generated from OHRC satellite image",
                SOURCE=self.image_path,
                **{k.upper(): str(v) for k, v in self.metadata.items()}
            )

        print(f"DEM saved to: {filename}")
        return filename

    def visualize_dem(self, save_plots=True):
        """Create visualizations of the DEM"""
        if self.dem is None:
            raise ValueError("DEM not generated yet. Call generate_dem() first.")

        print("Creating visualizations...")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))

        # Original image
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(self.image, cmap='gray')
        ax1.set_title('Original Lunar Image')
        ax1.axis('off')

        # DEM as 2D heatmap
        ax2 = fig.add_subplot(2, 3, 2)
        im = ax2.imshow(self.dem, cmap='terrain')
        ax2.set_title('Digital Elevation Model')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, label='Elevation (m)')

        # 3D surface plot
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        x = np.arange(0, self.dem.shape[1], 5)
        y = np.arange(0, self.dem.shape[0], 5)
        X, Y = np.meshgrid(x, y)
        Z = self.dem[::5, ::5]
        ax3.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8)
        ax3.set_title('3D Surface')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Elevation (m)')

        # Contour plot
        ax4 = fig.add_subplot(2, 3, 4)
        contours = ax4.contour(self.dem, levels=20, colors='black', alpha=0.6)
        ax4.contourf(self.dem, levels=20, cmap='terrain', alpha=0.8)
        ax4.set_title('Contour Map')
        ax4.axis('off')

        # Slope analysis
        ax5 = fig.add_subplot(2, 3, 5)
        grad_x = np.gradient(self.dem, axis=1)
        grad_y = np.gradient(self.dem, axis=0)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        ax5.imshow(slope, cmap='hot')
        ax5.set_title('Slope Analysis')
        ax5.axis('off')

        # Elevation histogram
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.hist(self.dem.flatten(), bins=50, alpha=0.7, color='skyblue')
        ax6.set_title('Elevation Distribution')
        ax6.set_xlabel('Elevation (m)')
        ax6.set_ylabel('Frequency')

        plt.tight_layout()

        if save_plots:
            plot_path = os.path.join(self.output_dir, "dem_visualization.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {plot_path}")

        plt.show()

    def generate_statistics(self):
        """Generate and save DEM statistics"""
        if self.dem is None:
            raise ValueError("DEM not generated yet. Call generate_dem() first.")

        stats = {
            'elevation_stats': {
                'min': float(np.min(self.dem)),
                'max': float(np.max(self.dem)),
                'mean': float(np.mean(self.dem)),
                'std': float(np.std(self.dem)),
                'median': float(np.median(self.dem)),
                'range': float(np.max(self.dem) - np.min(self.dem))
            },
            'image_info': {
                'dimensions': self.dem.shape,
                'total_pixels': int(self.dem.size)
            },
            'metadata': self.metadata
        }

        # Calculate slope statistics
        grad_x = np.gradient(self.dem, axis=1)
        grad_y = np.gradient(self.dem, axis=0)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)

        stats['slope_stats'] = {
            'min': float(np.min(slope)),
            'max': float(np.max(slope)),
            'mean': float(np.mean(slope)),
            'std': float(np.std(slope))
        }

        # Save statistics
        stats_path = os.path.join(self.output_dir, "dem_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Statistics saved to: {stats_path}")

        return stats


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Generate DEM from lunar OHRC satellite image')
    parser.add_argument('image_path', help='Path to the lunar image file')
    parser.add_argument('--metadata', help='Path to XML metadata file')
    parser.add_argument('--output', default='./dem_output', help='Output directory')
    parser.add_argument('--method', choices=['photometric', 'shape_from_shading', 'combined'],
                        default='combined', help='DEM generation method')

    args = parser.parse_args()

    # Create DEM generator
    dem_gen = LunarDEMGenerator(args.image_path, args.metadata, args.output)

    # Generate DEM
    dem = dem_gen.generate_dem(method=args.method)

    # Save DEM
    dem_gen.save_dem()

    # Generate visualizations
    dem_gen.visualize_dem()

    # Generate statistics
    dem_gen.generate_statistics()

    print("DEM generation complete!")


if __name__ == "__main__":
    main()