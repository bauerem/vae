import os
import scipy.misc
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

data_path = "/emnist-digits/"
output_dir = "output_dir/"

train_images_path = os.path.join(data_path,"emnist-digits-train-images-idx3-ubyte.gz")
train_labels_path = os.path.join(data_path,"emnist-digits-train-labels-idx1-ubyte.gz")

test_images_path = os.path.join(data_path,"emnist-digits-test-images-idx3-ubyte.gz")
test_labels_path = os.path.join(data_path,"emnist-digits-test-labels-idx1-ubyte.gz")



# load training images binary file
with open(train_images_path, 'rb') as f:
  train_images = extract_images(f)
# load training labels binary file
with open(train_labels_path, 'rb') as f:
  train_labels = extract_labels(f)


# load test images binary file
with open(test_images_path, 'rb') as f:
  test_images = extract_images(f)
# load test labels binary file
with open(test_labels_path, 'rb') as f:
  test_labels = extract_labels(f)



def save_class(save_dir, image, i, c):
    class_dir = os.path.join(output_dir, save_dir +'/class' + str(c))
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        scipy.misc.imsave(file_name,image)
    else:
        scipy.misc.imsave(file_name,image)

save_class = lambda c: (lambda save_dir, image, i: save_class(save_dir, image, i, c))

# dict instead of switch case or if else technique
class_label = {
        0: save_class(0),
        1: save_class(1),
        2: save_class(2),
        3: save_class(3),
        4: save_class(4),
        5: save_class(5),
        6: save_class(6),
        7: save_class(7),
        8: save_class(8),
        9: save_class(9)
        }
  
# saving training data
i = 0
num_images = len(train_images)
for i in range(0, num_images):
        
    image = train_images[i]
    image = image.transpose([1,2,0])
    image = image.reshape(28,28)
    label = train_labels[i]
    
    class_label[label]('train',image,i) # call dict as method
    i += 1
    
 
# saving test data
i = 0
num_images = len(test_images)
for i in range(0, num_images):
        
    image = test_images[i]
    image = image.transpose([1,2,0])
    image = image.reshape(28,28)
    label = test_labels[i]
    
    class_label[label]('test',image,i) # call dict as method
    i += 1