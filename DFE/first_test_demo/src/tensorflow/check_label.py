from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

# read
img = Image.open("example.jpg")
label = img.getexif().get(0x9286)          # 0x9286 = UserComment tag
print(label)

# # write
# meta = ImageFileDirectory_v2()
# meta[0x9286] = b"horse"                      # store "horse" as UserComment
# img.save("example_with_label.jpg", exif=meta.tobytes())
