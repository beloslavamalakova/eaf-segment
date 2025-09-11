import argparse

import cv2
import numpy as np


def get_center_of_mass(mask: np.ndarray) -> tuple[int, int]:
    """Calculate the center of mass of a binary mask."""
    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return (mask.shape[1] // 2, mask.shape[0] // 2)  # Default to center if no mass
    center_x = int(np.mean(x_indices))
    center_y = int(np.mean(y_indices))
    return (center_x, center_y)


def raycast(mask: np.ndarray, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Perform raycasting from the center in the given direction until hitting the mask boundary."""
    height, width = mask.shape
    x, y = center
    dx, dy = direction / np.linalg.norm(direction)

    while 0 <= int(x) < width and 0 <= int(y) < height:
        if mask[int(y), int(x)] == 0:  # Assuming mask is binary with 0 as background
            break
        x += dx
        y += dy

    return np.array([x, y])


def reverse_raycast(
    mask: np.ndarray, center: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Raycast from outside towards the center until hitting the mask boundary."""
    height, width = mask.shape
    x, y = center
    dx, dy = direction / np.linalg.norm(direction)

    # Start from a point far outside the image in the `direction`
    x += dx * max(width, height)
    y += dy * max(width, height)

    while (
        not (0 <= int(x) < width and 0 <= int(y) < height) or mask[int(y), int(x)] == 0
    ):
        x -= dx
        y -= dy

    return np.array([x, y])


class PolarMapping:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.center_of_mass = get_center_of_mass(image[:, :, 3])
        self.n_rays = 100
        self.boundary_points = self.convert_to_polar(self.center_of_mass, self.n_rays)

    def convert_to_polar(self, center: tuple[int, int], n_rays: int) -> np.ndarray:
        boundary_points = []
        for i in range(n_rays):
            radians = np.deg2rad(i * (360 / n_rays))
            direction = np.array([np.cos(radians), np.sin(radians)])
            boundary_point = reverse_raycast(
                self.image[:, :, 3], np.array(center), direction
            )
            boundary_points.append(boundary_point)

        return np.array(boundary_points)

    def pixel_at_polar(self, radians: float, radius_fraction: float) -> np.ndarray:
        """Get color at a specific polar coordinate.

        Interpolate between the boundary points to find the pixel color.
        """
        # Find the two boundary points to interpolate between
        angle_index_float = (radians / (2 * np.pi)) * (self.n_rays)
        angle_index = int(np.floor(angle_index_float))
        interpolation_factor = angle_index_float - angle_index
        assert -1e-9 < interpolation_factor < 1 + 1e-9, f"{interpolation_factor=}"

        p1 = self.boundary_points[angle_index % self.n_rays]
        p2 = self.boundary_points[(angle_index + 1) % self.n_rays]

        # Interpolate between p1 and p2
        interpolated_point = p1 + (p2 - p1) * (interpolation_factor)

        interpolated_point = (
            self.center_of_mass
            + (interpolated_point - self.center_of_mass) * radius_fraction
        )

        return interpolated_point

    def get_boundary_point(self, radians: float) -> np.ndarray:
        """Get the boundary point at a specific angle."""
        return self.pixel_at_polar(radians, 1.0)

    def polar_at_pixel(self, p: np.ndarray) -> tuple[float, float]:
        """Convert pixel coordinates to polar coordinates."""
        delta = p - self.center_of_mass
        angle = np.arctan2(delta[1], delta[0])
        boundary = self.get_boundary_point(angle)

        radius = np.linalg.norm(delta) / np.linalg.norm(boundary - self.center_of_mass)
        return (angle, radius)

    def contour(self) -> np.ndarray:
        """Return a grayscale image showing the contour."""
        contour_image = np.zeros(self.image.shape[:2], dtype=np.uint8)
        # Draw polygon using opencv
        pts = self.boundary_points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(contour_image, [pts], isClosed=True, color=255, thickness=1)
        return contour_image


def dion_stretch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Apply Dion's stretch algorithm to warp the source image to match the target image."""
    source_mapping = PolarMapping(source)

    # For debugging: visualize source contour
    # return source_mapping.contour()

    target_mapping = PolarMapping(target)

    height, width = target.shape[:2]
    output = np.zeros_like(target)

    for y in range(height):
        for x in range(width):
            angle, radius = target_mapping.polar_at_pixel(np.array([x, y]))

            if radius > 1.0:
                output[y, x] = 0
                continue

            # For debugging: visualize polar coordinates
            # output[y, x, 0] = radius * 255
            # output[y, x, 1] = angle / (np.pi * 2) * 255
            # output[y, x, 3] = 255

            source_point = source_mapping.pixel_at_polar(angle, radius)

            sx, sy = int(source_point[0]), int(source_point[1])
            if 0 <= sx < source.shape[1] and 0 <= sy < source.shape[0]:
                output[y, x] = source[sy, sx]

    # Apply alpha mask of target to output
    output[:, :, 3] = target[:, :, 3]

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dion's Stretch Image Warping")
    parser.add_argument("source", type=str, help="Path to the source image")
    parser.add_argument("target", type=str, help="Path to the target image")
    parser.add_argument("output", type=str, help="Path to save the output image")
    args = parser.parse_args()

    source_img = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)
    target_img = cv2.imread(args.target, cv2.IMREAD_UNCHANGED)

    if source_img is None or target_img is None:
        raise ValueError("Could not load source or target image.")

    # Ensure both images have an alpha channel
    assert source_img.shape[2] == 4, "Source image must have an alpha channel"
    assert target_img.shape[2] == 4, "Target image must have an alpha channel"

    output_img = dion_stretch(source_img, target_img)
    cv2.imwrite(args.output, output_img)
