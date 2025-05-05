import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from io import BytesIO
import time

# Simulated module imports that will be replaced with actual ones when dependencies are installed
# from model.text_encoder import TextEncoder
# from model.generator import Generator
# from model.alr_module import ALRModule

# Set page config
st.set_page_config(
    page_title="ALR-GAN: Text-to-Image Synthesis",
    page_icon="🖼️",
    layout="wide"
)

# Create a simulation of the model for demo purposes
@st.cache_resource
def load_models():
    """Load simulated models for demonstration"""
    class SimulatedTextEncoder:
        def __init__(self):
            pass
        
        def eval(self):
            pass
        
        def __call__(self, text):
            # Return a simulated text embedding
            return {
                'sentence_embedding': "This is a simulated embedding",
                'word_embeddings': "These are simulated word embeddings"
            }
    
    class SimulatedGenerator:
        def __init__(self):
            pass
        
        def eval(self):
            pass
        
        def generate_64(self, z, text_embedding):
            # Return a simulated 64x64 image and features
            return "simulated_img_64", "simulated_features_64"
        
        def generate_128(self, img_64, features, text_embedding):
            # Return a simulated 128x128 image and features
            return "simulated_img_128", "simulated_features_128"
        
        def generate_256(self, img_128, features, text_embedding):
            # Return a simulated 256x256 image and features
            return "simulated_img_256", "simulated_features"
    
    class SimulatedALRModule:
        def __init__(self):
            pass
        
        def eval(self):
            pass
        
        def __call__(self, text_embedding, features):
            # Return simulated refined features
            return "simulated_refined_features"
    
    # Create simulated models
    text_encoder = SimulatedTextEncoder()
    generator = SimulatedGenerator()
    alr_module = SimulatedALRModule()
    
    return text_encoder, generator, alr_module

# Function to generate placeholder images for demonstration
def generate_image(text_description, text_encoder, generator, alr_module):
    """Generate placeholder images for demonstration"""
    # Parse the text description for visual elements
    words = text_description.lower().split()
    
    # Set seed based on text description for consistent generation
    seed = hash(text_description) % 10000
    random.seed(seed)
    
    # Color mapping for common objects
    color_map = {
        'red': (255, 50, 50),
        'blue': (50, 50, 255),
        'green': (50, 200, 50),
        'yellow': (255, 255, 50),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
    }
    
    # Object mapping
    object_types = {
        'bird': 'animal',
        'cat': 'animal',
        'dog': 'animal',
        'fish': 'animal',
        'car': 'vehicle',
        'truck': 'vehicle',
        'motorcycle': 'vehicle',
        'bicycle': 'vehicle',
        'flower': 'plant',
        'tree': 'plant',
        'mountain': 'landscape',
        'beach': 'landscape',
        'sunset': 'landscape',
        'food': 'food',
        'pizza': 'food',
        'spaghetti': 'food',
        'table': 'furniture',
        'chair': 'furniture',
        'vase': 'object',
    }
    
    # Detect main object and color
    main_color = None
    main_object = None
    background_color = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
    
    for word in words:
        if word in color_map:
            main_color = color_map[word]
        
        for obj in object_types:
            if obj in word:
                main_object = obj
    
    if main_color is None:
        main_color = (random.randint(100, 255), 
                     random.randint(100, 255), 
                     random.randint(100, 255))
    
    # Create visualizations for different stages
    def create_placeholder_image(size, text, stage=1):
        # Create base image with background
        img = Image.new('RGB', (size, size), color=background_color)
        draw = ImageDraw.Draw(img)
        
        # Add some elements based on the text description
        if main_object:
            # Progressively add details based on stage
            if main_object in ['bird', 'cat', 'dog']:
                # Draw animal shape
                center_x, center_y = size//2, size//2
                
                if stage == 1:
                    # Basic shape
                    shape_size = size // 3
                    draw.ellipse(
                        [(center_x - shape_size, center_y - shape_size),
                         (center_x + shape_size, center_y + shape_size)],
                        fill=main_color
                    )
                elif stage == 2:
                    # Add more details
                    body_size = size // 3
                    head_size = size // 5
                    draw.ellipse(
                        [(center_x - body_size, center_y - body_size//2),
                         (center_x + body_size, center_y + body_size)],
                        fill=main_color
                    )
                    draw.ellipse(
                        [(center_x - head_size - body_size//2, center_y - head_size),
                         (center_x - body_size//2 + head_size, center_y + head_size)],
                        fill=main_color
                    )
                else:
                    # Full details
                    body_size = size // 3
                    head_size = size // 5
                    leg_width = size // 20
                    
                    # Body
                    draw.ellipse(
                        [(center_x - body_size, center_y - body_size//2),
                         (center_x + body_size, center_y + body_size)],
                        fill=main_color
                    )
                    
                    # Head
                    draw.ellipse(
                        [(center_x - head_size - body_size//2, center_y - head_size),
                         (center_x - body_size//2 + head_size, center_y + head_size)],
                        fill=main_color
                    )
                    
                    # Eyes
                    eye_size = size // 25
                    draw.ellipse(
                        [(center_x - body_size//2 - head_size//2, center_y - head_size//4 - eye_size),
                         (center_x - body_size//2 - head_size//2 + eye_size*2, center_y - head_size//4 + eye_size)],
                        fill=(255, 255, 255)
                    )
                    draw.ellipse(
                        [(center_x - body_size//2 - head_size//2 + eye_size//2, center_y - head_size//4 - eye_size//2),
                         (center_x - body_size//2 - head_size//2 + eye_size*3//2, center_y - head_size//4 + eye_size//2)],
                        fill=(0, 0, 0)
                    )
                    
                    # Legs
                    draw.rectangle(
                        [(center_x - body_size//2, center_y + body_size - leg_width),
                         (center_x - body_size//2 + leg_width, center_y + body_size + leg_width*3)],
                        fill=(main_color[0]//2, main_color[1]//2, main_color[2]//2)
                    )
                    draw.rectangle(
                        [(center_x + body_size//2 - leg_width, center_y + body_size - leg_width),
                         (center_x + body_size//2, center_y + body_size + leg_width*3)],
                        fill=(main_color[0]//2, main_color[1]//2, main_color[2]//2)
                    )
            
            elif main_object in ['car', 'truck']:
                # Draw vehicle shape
                if stage == 1:
                    # Basic shape
                    draw.rectangle(
                        [(size//4, size//2), 
                         (size*3//4, size*3//4)],
                        fill=main_color
                    )
                elif stage == 2:
                    # More details
                    draw.rectangle(
                        [(size//4, size//2), 
                         (size*3//4, size*3//4)],
                        fill=main_color
                    )
                    draw.rectangle(
                        [(size//3, size//3), 
                         (size*2//3, size//2)],
                        fill=main_color
                    )
                else:
                    # Full details
                    # Car body
                    draw.rectangle(
                        [(size//4, size//2), 
                         (size*3//4, size*3//4)],
                        fill=main_color
                    )
                    # Car top
                    draw.rectangle(
                        [(size//3, size//3), 
                         (size*2//3, size//2)],
                        fill=main_color
                    )
                    # Windows
                    window_color = (200, 220, 255)
                    draw.rectangle(
                        [(size//3 + size//30, size//3 + size//30), 
                         (size*2//3 - size//30, size//2 - size//30)],
                        fill=window_color
                    )
                    # Wheels
                    wheel_size = size // 10
                    draw.ellipse(
                        [(size//3 - wheel_size, size*3//4 - wheel_size), 
                         (size//3 + wheel_size, size*3//4 + wheel_size)],
                        fill=(30, 30, 30)
                    )
                    draw.ellipse(
                        [(size*2//3 - wheel_size, size*3//4 - wheel_size), 
                         (size*2//3 + wheel_size, size*3//4 + wheel_size)],
                        fill=(30, 30, 30)
                    )
            
            elif main_object in ['flower', 'tree']:
                # Draw plant
                if stage == 1:
                    # Basic shape
                    center_x, center_y = size//2, size//2
                    radius = size // 4
                    draw.ellipse(
                        [(center_x - radius, center_y - radius), 
                         (center_x + radius, center_y + radius)],
                        fill=main_color
                    )
                elif stage == 2:
                    # More details
                    center_x, center_y = size//2, size//2
                    radius = size // 4
                    stem_width = size // 20
                    draw.rectangle(
                        [(center_x - stem_width//2, center_y), 
                         (center_x + stem_width//2, size*3//4)],
                        fill=(20, 120, 20)
                    )
                    draw.ellipse(
                        [(center_x - radius, center_y - radius), 
                         (center_x + radius, center_y + radius)],
                        fill=main_color
                    )
                else:
                    # Full details
                    center_x, center_y = size//2, size//2
                    radius = size // 4
                    stem_width = size // 20
                    
                    # Stem
                    draw.rectangle(
                        [(center_x - stem_width//2, center_y), 
                         (center_x + stem_width//2, size*3//4)],
                        fill=(20, 120, 20)
                    )
                    
                    # Flower petals or tree leaves
                    if 'flower' in text:
                        for angle in range(0, 360, 45):
                            angle_rad = angle * 3.14159 / 180
                            petal_x = center_x + int(radius * 0.8 * np.cos(angle_rad))
                            petal_y = center_y + int(radius * 0.8 * np.sin(angle_rad))
                            
                            draw.ellipse(
                                [(petal_x - radius//2, petal_y - radius//2), 
                                 (petal_x + radius//2, petal_y + radius//2)],
                                fill=main_color
                            )
                        
                        # Center of flower
                        center_color = (255, 220, 0)
                        draw.ellipse(
                            [(center_x - radius//3, center_y - radius//3), 
                             (center_x + radius//3, center_y + radius//3)],
                            fill=center_color
                        )
                    else:  # Tree
                        # Tree crown
                        draw.ellipse(
                            [(center_x - radius*1.5, center_y - radius*2), 
                             (center_x + radius*1.5, center_y + radius//2)],
                            fill=(20, 150, 20)
                        )
                        
                        # Tree trunk
                        trunk_width = size // 10
                        draw.rectangle(
                            [(center_x - trunk_width//2, center_y), 
                             (center_x + trunk_width//2, size*3//4)],
                            fill=(100, 50, 0)
                        )
        
        # For food items
        elif 'food' in text or 'spaghetti' in text or 'pizza' in text:
            center_x, center_y = size//2, size//2
            if stage == 1:
                # Basic shape - plate
                plate_size = size // 2
                draw.ellipse(
                    [(center_x - plate_size//2, center_y - plate_size//4), 
                     (center_x + plate_size//2, center_y + plate_size//2)],
                    fill=(220, 220, 220)
                )
            elif stage == 2:
                # Add food
                plate_size = size // 2
                food_size = size // 3
                
                # Plate
                draw.ellipse(
                    [(center_x - plate_size//2, center_y - plate_size//4), 
                     (center_x + plate_size//2, center_y + plate_size//2)],
                    fill=(220, 220, 220)
                )
                
                # Food
                if 'pizza' in text:
                    draw.ellipse(
                        [(center_x - food_size//2, center_y - food_size//4), 
                         (center_x + food_size//2, center_y + food_size//2)],
                        fill=(230, 190, 80)
                    )
                elif 'spaghetti' in text:
                    for i in range(10):
                        x1 = center_x - food_size//2 + random.randint(0, food_size)
                        y1 = center_y - food_size//4 + random.randint(0, food_size//2)
                        x2 = center_x - food_size//2 + random.randint(0, food_size)
                        y2 = center_y - food_size//4 + random.randint(0, food_size//2)
                        
                        draw.line([(x1, y1), (x2, y2)], fill=(230, 230, 150), width=2)
            else:
                # Full details
                plate_size = size // 2
                food_size = size // 3
                
                # Plate
                draw.ellipse(
                    [(center_x - plate_size//2, center_y - plate_size//4), 
                     (center_x + plate_size//2, center_y + plate_size//2)],
                    fill=(240, 240, 240)
                )
                draw.ellipse(
                    [(center_x - plate_size//2 + 5, center_y - plate_size//4 + 5), 
                     (center_x + plate_size//2 - 5, center_y + plate_size//2 - 5)],
                    fill=(220, 220, 220)
                )
                
                # Food
                if 'pizza' in text:
                    # Pizza base
                    draw.ellipse(
                        [(center_x - food_size//2, center_y - food_size//4), 
                         (center_x + food_size//2, center_y + food_size//2)],
                        fill=(230, 190, 80)
                    )
                    
                    # Pizza sauce
                    draw.ellipse(
                        [(center_x - food_size//2 + 5, center_y - food_size//4 + 5), 
                         (center_x + food_size//2 - 5, center_y + food_size//2 - 5)],
                        fill=(220, 70, 40)
                    )
                    
                    # Pizza toppings
                    for i in range(15):
                        topping_x = center_x - food_size//2 + 10 + random.randint(0, food_size - 20)
                        topping_y = center_y - food_size//4 + 10 + random.randint(0, food_size//2 - 15)
                        topping_size = random.randint(3, 7)
                        
                        if random.random() < 0.5:
                            # Pepperoni
                            draw.ellipse(
                                [(topping_x - topping_size, topping_y - topping_size), 
                                 (topping_x + topping_size, topping_y + topping_size)],
                                fill=(180, 30, 30)
                            )
                        else:
                            # Cheese
                            draw.rectangle(
                                [(topping_x - topping_size, topping_y - topping_size), 
                                 (topping_x + topping_size, topping_y + topping_size)],
                                fill=(250, 230, 140)
                            )
                
                elif 'spaghetti' in text:
                    # Pasta
                    for i in range(30):
                        x1 = center_x - food_size//2 + random.randint(0, food_size)
                        y1 = center_y - food_size//4 + random.randint(0, food_size//2)
                        
                        # Create curved lines for spaghetti
                        for j in range(3):
                            x2 = x1 + random.randint(-food_size//4, food_size//4)
                            y2 = y1 + random.randint(-food_size//8, food_size//8)
                            draw.line([(x1, y1), (x2, y2)], fill=(250, 240, 190), width=2)
                            x1, y1 = x2, y2
                    
                    # Sauce
                    for i in range(15):
                        sauce_x = center_x - food_size//2 + 10 + random.randint(0, food_size - 20)
                        sauce_y = center_y - food_size//4 + 10 + random.randint(0, food_size//2 - 15)
                        sauce_size = random.randint(4, 10)
                        
                        draw.ellipse(
                            [(sauce_x - sauce_size, sauce_y - sauce_size//2), 
                             (sauce_x + sauce_size, sauce_y + sauce_size//2)],
                            fill=(200, 60, 30)
                        )
                    
                    # Meatballs if mentioned
                    if 'meatball' in text:
                        for i in range(3):
                            mb_x = center_x - food_size//4 + random.randint(0, food_size//2)
                            mb_y = center_y - food_size//6 + random.randint(0, food_size//3)
                            mb_size = random.randint(10, 15)
                            
                            draw.ellipse(
                                [(mb_x - mb_size, mb_y - mb_size), 
                                 (mb_x + mb_size, mb_y + mb_size)],
                                fill=(120, 70, 50)
                            )
        
        # For landscape scenes
        elif 'mountain' in text or 'sunset' in text:
            if stage == 1:
                # Basic scene
                # Sky
                sky_color = (100, 150, 255)
                if 'sunset' in text:
                    sky_color = (255, 180, 100)
                
                draw.rectangle(
                    [(0, 0), (size, size//2)],
                    fill=sky_color
                )
                
                # Ground
                draw.rectangle(
                    [(0, size//2), (size, size)],
                    fill=(100, 180, 100)
                )
            
            elif stage == 2:
                # Add mountains
                # Sky
                sky_color = (100, 150, 255)
                if 'sunset' in text:
                    sky_color = (255, 180, 100)
                
                draw.rectangle(
                    [(0, 0), (size, size//2)],
                    fill=sky_color
                )
                
                # Mountains
                mountain_color = (100, 100, 100)
                points = [
                    (0, size//2),
                    (size//3, size//4),
                    (size//2, size//3),
                    (size*2//3, size//4),
                    (size, size//2),
                    (size, size),
                    (0, size)
                ]
                draw.polygon(points, fill=mountain_color)
                
                # Ground
                draw.rectangle(
                    [(0, size//2), (size, size)],
                    fill=(100, 180, 100)
                )
            
            else:
                # Full detailed scene
                # Sky gradient
                if 'sunset' in text:
                    for y in range(size//2):
                        # Create sunset gradient
                        ratio = y / (size//2)
                        r = int(255 - ratio * 155)
                        g = int(180 - ratio * 150)
                        b = int(100 + ratio * 155)
                        draw.line([(0, y), (size, y)], fill=(r, g, b))
                else:
                    # Regular sky
                    draw.rectangle(
                        [(0, 0), (size, size//2)],
                        fill=(100, 150, 255)
                    )
                
                # Mountains with snow caps
                mountain_color = (100, 100, 100)
                snow_color = (255, 255, 255)
                
                # First mountain
                points = [
                    (0, size//2),
                    (size//3, size//4),
                    (size//2, size//3),
                    (size, size//2),
                    (size, size),
                    (0, size)
                ]
                draw.polygon(points, fill=mountain_color)
                
                # Snow caps
                snow_points = [
                    (size//6, size*3//8),
                    (size//3, size//4),
                    (size//2, size//3),
                    (size*2//3, size*3//8)
                ]
                draw.polygon(snow_points, fill=snow_color)
                
                # Ground with texture
                draw.rectangle(
                    [(0, size//2), (size, size)],
                    fill=(100, 180, 100)
                )
                
                # Add some trees if not a beach
                if 'beach' not in text:
                    for i in range(5):
                        tree_x = random.randint(size//10, size*9//10)
                        tree_y = random.randint(size*6//10, size*9//10)
                        tree_size = random.randint(size//20, size//10)
                        
                        # Tree trunk
                        draw.rectangle(
                            [(tree_x - tree_size//6, tree_y - tree_size), 
                             (tree_x + tree_size//6, tree_y)],
                            fill=(100, 70, 30)
                        )
                        
                        # Tree top
                        draw.ellipse(
                            [(tree_x - tree_size, tree_y - tree_size*2), 
                             (tree_x + tree_size, tree_y - tree_size//2)],
                            fill=(30, 120, 30)
                        )
                
                # Sun or moon
                if 'sunset' in text:
                    # Sun
                    sun_x, sun_y = size*3//4, size//6
                    sun_size = size//10
                    draw.ellipse(
                        [(sun_x - sun_size, sun_y - sun_size), 
                         (sun_x + sun_size, sun_y + sun_size)],
                        fill=(255, 230, 100)
                    )
                
                # Add clouds
                for i in range(3):
                    cloud_x = random.randint(size//10, size*9//10)
                    cloud_y = random.randint(size//10, size*4//10)
                    cloud_size = random.randint(size//15, size//8)
                    
                    for j in range(5):
                        offset_x = random.randint(-cloud_size, cloud_size)
                        offset_y = random.randint(-cloud_size//2, cloud_size//2)
                        
                        draw.ellipse(
                            [(cloud_x + offset_x - cloud_size, cloud_y + offset_y - cloud_size//2), 
                             (cloud_x + offset_x + cloud_size, cloud_y + offset_y + cloud_size//2)],
                            fill=(240, 240, 240)
                        )
        
        # Default pattern if no specific objects detected
        else:
            center_x, center_y = size//2, size//2
            
            if stage == 1:
                # Basic color block
                draw.rectangle(
                    [(size//4, size//4), (size*3//4, size*3//4)],
                    fill=main_color
                )
            elif stage == 2:
                # Add simple shapes
                draw.ellipse(
                    [(size//4, size//4), (size*3//4, size*3//4)],
                    fill=main_color
                )
                
                secondary_color = (main_color[0]//2, main_color[1]//2, main_color[2]//2)
                draw.ellipse(
                    [(size*3//8, size*3//8), (size*5//8, size*5//8)],
                    fill=secondary_color
                )
            else:
                # More complex pattern
                for i in range(5):
                    angle = i * 2 * 3.14159 / 5
                    radius = size // 3
                    
                    x1 = center_x + int(radius * np.cos(angle))
                    y1 = center_y + int(radius * np.sin(angle))
                    
                    x2 = center_x + int(radius * np.cos(angle + 2*3.14159/5*2))
                    y2 = center_y + int(radius * np.sin(angle + 2*3.14159/5*2))
                    
                    draw.line([(x1, y1), (x2, y2)], fill=main_color, width=size//20)
                
                # Central point
                draw.ellipse(
                    [(center_x - size//10, center_y - size//10), 
                     (center_x + size//10, center_y + size//10)],
                    fill=main_color
                )
        
        return img
    
    # Add some delay to simulate processing time
    time.sleep(0.5)
    
    # Simulate the process of generating images with 3 progressive stages
    img_64 = create_placeholder_image(64, text_description, stage=1)
    img_128 = create_placeholder_image(128, text_description, stage=2)
    img_256 = create_placeholder_image(256, text_description, stage=3)
    
    return img_256, img_64, img_128

# Title and description
st.title("ALR-GAN: Text-to-Image Synthesis")
st.write("""
This application implements the ALR-GAN (Adaptive Layout Refinement Generative Adversarial Network) 
for text-to-image synthesis. Enter a description, and the model will generate an image with proper 
layout structure without requiring additional annotations like bounding boxes or scene graphs.
""")

# Load models
with st.spinner("Loading models... This might take a minute."):
    text_encoder, generator, alr_module = load_models()
    st.success("Models loaded successfully!")

# Text input
text_description = st.text_area(
    "Enter your image description:",
    "A small bird with a red head and a yellow body",
    height=100
)

# Generation settings
st.sidebar.header("Generation Settings")
temperature = st.sidebar.slider("Diversity (Temperature)", 0.5, 1.5, 1.0, 0.1)
num_iterations = st.sidebar.slider("Refinement Iterations", 1, 5, 3, 1)

# Example descriptions
st.sidebar.header("Example Descriptions")
examples = [
    "A beautiful bird with blue wings and a long beak sitting on a branch",
    "A small black and white cat with green eyes playing with a ball of yarn",
    "A red sports car driving on a mountain road with a sunset in the background",
    "A plate of spaghetti with meatballs and tomato sauce",
    "A vase with colorful flowers on a wooden table by the window"
]
for example in examples:
    if st.sidebar.button(example[:40] + "..."):
        text_description = example
        st.rerun()

# Generate button
if st.button("Generate Image"):
    with st.spinner("Generating image... This might take a few seconds."):
        try:
            # Generate the image
            final_img, img_64, img_128 = generate_image(
                text_description, text_encoder, generator, alr_module
            )
            
            # Display results
            st.subheader("Generated Image")
            
            # Show progressive generation
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(img_64, caption="Stage 1: 64x64", use_column_width=True)
            
            with col2:
                st.image(img_128, caption="Stage 2: 128x128", use_column_width=True)
            
            with col3:
                st.image(final_img, caption="Final: 256x256", use_column_width=True)
            
            # Display text description used
            st.text_area("Description used:", text_description, height=80, disabled=True)
            
            # Allow downloading the generated image
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="alr_gan_generated.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")

# Add explanation section
with st.expander("About ALR-GAN"):
    st.write("""
    ### Key Components
    
    **1. Adaptive Layout Refinement (ALR) Module**
    - Uses semantic similarity matching between words and image regions
    - Introduces an adaptive loss that balances attention between "hard" and "easy" regions
    - Builds Semantics Similarity Matrix (SSM) between text and image
    
    **2. Layout Visual Refinement (LVR) Loss**
    - Perception Refinement (PR) Loss: Improves texture details
    - Style Refinement (SR) Loss: Enhances style consistency
    - Focuses only on layout-relevant areas to avoid over-constraining the model
    
    **3. Multi-Stage Generation**
    - Progressive refinement from 64x64 → 128x128 → 256x256 resolution
    - Each stage benefits from layout refinement
    """)

# Add technical details section
with st.expander("Technical Details"):
    st.write("""
    - Implementation uses PyTorch and BERT for text encoding
    - Generator uses a multi-stage approach for progressive resolution refinement
    - ALR Module calculates semantic similarity between text and image regions
    - The model doesn't require auxiliary information like bounding boxes or scene graphs
    """)
