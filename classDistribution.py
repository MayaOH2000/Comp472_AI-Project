import os
import matplotlib.pyplot as plt


#Count number of images in each folder
def count_images_in_classes(directory):
    class_counts = {}

    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        
        if os.path.isdir(class_path):
            image_count = len([file for file in os.listdir(class_path) if file.endswith(('.jpg', '.png'))])
            class_counts[class_name] = image_count

    return class_counts

#Bar graph parameters and values
def plot_bar_graph(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.bar(classes, counts, color='blue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Image Counts in Each Class')
    plt.show()

if __name__ == "__main__":
    #File path with all class folders
    image_directory = 'Dataset\\train'

    #display each classe count in terminal consol
    class_counts = count_images_in_classes(image_directory)
    print("Image counts in each class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")

    #diaplsy bar graph with image count
    plot_bar_graph(class_counts)
