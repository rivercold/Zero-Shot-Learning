import numpy as np

def main():
    # Part 1: pick out 40 unseen classes.
    unseen_set = set()

    total_number = 200
    unseen_number = 40
    left_number = total_number

    for i in xrange(1, total_number + 1):
        prob = unseen_number / float(left_number)
        rand = np.random.random()

        if rand < prob:
            # hit
            unseen_set.add(i)
            unseen_number -= 1

        left_number -= 1

    with open('unseen_classes_temp.txt', 'w') as f:
        unseen_list = sorted(list(unseen_set))
        for num in unseen_list:
            f.write(str(num) + '\n')


    # Part 2: assign each image to training set or test set. For seen classes, 0.8 for training and 0.2 for test. For unseen classes, 1 for test.
    total_image_number = 11788
    training_proportion = 0.8
    image_class_map = dict()
    with open('images.txt', 'r') as f:
        for line in f:
            contents = line.split(' ')
            image_idx = int(contents[0])
            class_idx = int(contents[1].split('.')[0])
            image_class_map[image_idx] = class_idx

    with open('train_test_split_temp.txt', 'w') as f:
        for j in xrange(1, total_image_number + 1):
            if image_class_map[j] not in unseen_set:
                if np.random.random() < training_proportion:
                    f.write(str(j) + ' 1\n')
                else:
                    f.write(str(j) + ' 0\n')
            else:
                f.write(str(j) + ' 0\n')




if __name__ == '__main__':
    np.random.seed(1)

    main()