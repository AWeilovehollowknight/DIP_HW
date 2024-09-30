import gradio as gr
import cv2
import numpy as np

#Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])
# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    image = np.array(image)

    # 计算填充大小，以避免边界问题
    pad_size = min(image.shape[0], image.shape[1]) // 2
    padded_image = np.full((image.shape[0] + 2 * pad_size, image.shape[1] + 2 * pad_size, 3), 255, dtype=np.uint8)

    # 将原始图像放入填充图像的中心
    padded_image[pad_size:pad_size + image.shape[0], pad_size:pad_size + image.shape[1]] = image

    # 获取填充后图像的高度和宽度
    height, width = padded_image.shape[:2]
    center = (width / 2, height / 2)

    # 创建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, scale)

    # 添加平移
    rotation_matrix[0, 2] += translation_x
    rotation_matrix[1, 2] += translation_y

    # 仿射变换
    transformed_image = cv2.warpAffine(padded_image, rotation_matrix, (width, height), borderValue=(255, 255, 255))

    # 水平翻转
    if flip_horizontal:
        transformed_image = cv2.flip(transformed_image, 1)

    return transformed_image
# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
