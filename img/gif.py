from PIL import Image

# Load all PNGs (sorted by name)
frames = [Image.open(f"frame{i}.png") for i in range(85)]

# Save as animated GIF
frames[0].save(
    "animation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,  # ms per frame
    loop=0,  # 0 = infinite loop
)
