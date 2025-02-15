import os

def convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    """Convert bounding box coordinates to YOLO format."""
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def process_annotation_file(input_file, output_dir, img_width, img_height):
    """Process a single annotation file and save YOLO format files for each frame."""
    with open(input_file, 'r') as f:
        lines = f.readlines()

    frame_annotations = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue  # Skip lines that don't have enough elements
        try:
            frame_id = int(parts[5])
        except ValueError:
            continue  # Skip lines where frame_id is not an integer
        x1, y1, x2, y2 = map(int, parts[1:5])
        class_id = 0  # Assuming class_id is 0 for all players

        if frame_id not in frame_annotations:
            frame_annotations[frame_id] = []

        x_center, y_center, width, height = convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height)
        frame_annotations[frame_id].append(f"{class_id} {x_center} {y_center} {width} {height}")

    for frame_id, annotations in frame_annotations.items():
        output_file = os.path.join(output_dir, f"{frame_id}.txt")
        with open(output_file, 'w') as f:
            f.write("\n".join(annotations))

def process_all_annotations(input_dir, output_dir, img_width, img_height):
    """Process all annotation files in the input directory."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                input_file = os.path.join(root, file)
                process_annotation_file(input_file, output_dir, img_width, img_height)

# Example usage
input_dir = '/path/to/txt/annotations'
output_dir = 'output/yolo_annotations'
img_width = 1280  # Replace with actual image width
img_height = 720  # Replace with actual image height

os.makedirs(output_dir, exist_ok=True)
process_all_annotations(input_dir, output_dir, img_width, img_height)