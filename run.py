#! /usr/bin/env python3
from pytocl.main import main
from my_driver import MyDriver


if __name__ == '__main__':

    for i in range(1, 51):
        print(f'Driver {i}:')
        main(MyDriver(logdata=False))