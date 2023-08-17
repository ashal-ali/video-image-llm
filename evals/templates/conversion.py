def read_second_column(filename):
    second_column_list = []
    with open(filename, 'r') as file:
        for line in file:
            _, second_column = line.strip().split(' ', 1)
            second_column_list.append(second_column)
    return second_column_list

if __name__ == "__main__":
    filename = "imagenet_classnames.txt"
    result = read_second_column(filename)
    print(result)