if __name__ == '__main__':
    sum = 0
    for i in range(1, 16):
        sum += 400 * pow((1 + 0.15), -i)
        print(f"t = {i} : {sum}")
    print(sum)

    a1 = 1 / 1.15
    q = 1 / 1.15
    n = 15

    sn = a1 * (1 - pow(q, n)) / (1 - q)
    print(400 * sn)
