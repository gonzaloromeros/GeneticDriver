#! /usr/bin/env python3
from pytocl.main import main
from my_driver import MyDriver


if __name__ == '__main__':
    for g in range(1, 11):
        print(f'--Generation {g}--')

        for i in range(1, 4):
            print(f'Driver {i}:')
            main(MyDriver(logdata=False, generation=g, n=i))
