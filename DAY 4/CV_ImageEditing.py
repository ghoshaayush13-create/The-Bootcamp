import cv2

# Read the image
image = cv2.imread(r"C:\Users\Aayush\Downloads\Image.png")
if image is None:
    print("Error: Image not found!")
    exit()

# Display original image
cv2.imshow("Original Image", image)

# Resized Iamge
resized = cv2.resize(image, (500, 500))
cv2.imshow("Resized Image", resized)

# Flip the Image
img_flip=cv2.flip(image,0)
cv2.imshow("Flipped Image",img_flip)

# Crop the Image
img_crop=image[100:300,200:500]
cv2.imshow("Cropped Image",img_crop)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow("Blurred Image", blur)

# Edge Detection
edges = cv2.Canny(image, 100, 200)
cv2.imshow("Edge Detection", edges)

# Adjust brightness
bright = cv2.convertScaleAbs(image, alpha=1.2, beta=40)
cv2.imshow("Brightened Image", bright)

# Save edited images
cv2.imwrite("resized.jpg", resized)
cv2.imwrite("grayscale.jpg", gray)
cv2.imwrite("blurred.jpg", blur)
cv2.imwrite("edges.jpg", edges)
cv2.imwrite("brightened.jpg", bright)

print("Image editing completed successfully!")
print("Edited images have been saved.")

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()