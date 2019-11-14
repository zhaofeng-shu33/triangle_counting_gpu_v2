import struct
import os

BUILD_DIR = os.environ.get('BUILD_DIR', 'build')

if __name__ == '__main__':
    f = open(os.path.join(BUILD_DIR, 'test_io.bin'), 'wb')
    f.write(struct.pack('6I',0,1,2,0,1,2))
    f.close()
    f = open(os.path.join(BUILD_DIR, 'test_io_false.bin'), 'wb')
    f.write(struct.pack('5I',0,1,1,2,0))
    f.close()
    f = open(os.path.join(BUILD_DIR, 'test_io_nvgraph.bin'), 'wb')
    f.write(struct.pack('16I',1,0,2,1,3,1,3,2,4,2,4,3,5,4,5,3))
    f = open(os.path.join(BUILD_DIR, 'test_zero_degree.bin'), 'wb')
    f.write(struct.pack('10I',0,1,2,0,1,2,2,0,2,0))
