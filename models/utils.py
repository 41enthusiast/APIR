import base64
from io import BytesIO
import predict_mask

def image_to_base64(image):
    """Converts a PIL image to a base64 data URI."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"
