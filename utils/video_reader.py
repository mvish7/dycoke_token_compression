from decord import VideoReader
import torchvision.transforms as transforms


def read_video_as_images(video_path,
                         fps=1.0,
                         max_frames=None,
                         sampling_factor=4):
    """
    Read video file and return as multiple PIL images stacked together.

    Args:
        video_path (str): Path to the video file
        fps (float): Target frames per second (default: 1.0)
        max_frames (int): Maximum number of frames to extract (default: None)
        target_size (tuple): Target size for images (width, height)

    Returns:
        list: List of PIL Images
    """
    print(f"Reading video: {video_path}")

    # Try using decord first (more efficient)

    # Use decord for video reading
    vr = VideoReader(video_path)

    # Calculate frame indices based on fps
    sampling_factor = sampling_factor

    # Get frame indices
    frame_indices = list(range(0, len(vr), sampling_factor))

    # Limit frames if max_frames is specified
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]

    # Read frames
    frames = vr.get_batch(frame_indices).asnumpy()

    # Convert to PIL images
    images = []

    for frame_id in range(len(frames)):
        # Convert from RGB to PIL
        frame = frames[frame_id]
        frame_pil = transforms.ToPILImage()(frame)
        images.append(frame_pil)

    print(f"Extracted {len(images)} frames from video")
    return images