#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#define DATA_MIN -6.00
#define DATA_MAX 6.00
#define CAT_MIN -1.0
#define CAT_MAX 1.0
#define NORM_MIN -3.0
#define NORM_MAX 3.0

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));
    int fd = open(argv[1], O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
    uint32_t number = strtoul(argv[2], NULL, 0);
    ftruncate(fd, (number*(sizeof(float)*102))+sizeof(uint32_t));
    write(fd, &number, 4);
    float *buffer = malloc(102*sizeof(float));
    if(!buffer) {
        printf("Could not allocate buffer!\n");
        exit(1);
    }
    for(int i = 0; i < number; i++) {
        // Cat Attribute
        buffer[0] = ((CAT_MAX - CAT_MIN) * ((float)rand() / RAND_MAX)) + CAT_MIN;
        // Norm Attribute
        buffer[1] = ((NORM_MAX - NORM_MIN) * ((float)rand() / RAND_MAX)) + NORM_MIN;
        for(int j = 2; j < 102; j++) {
            buffer[j] = ((DATA_MAX - DATA_MIN) * ((float)rand() / RAND_MAX)) + DATA_MIN;
        }
        if(write(fd, buffer, 102*sizeof(float)) == -1) {
            printf("Something went wrong while writing the file!\n");
            break;
        }

        if (i % (number / 10) == 0) {
            printf("%u%%\n", i / (number / 10) * 10);
        }
    }
    fsync(fd);
    close(fd);
    printf("closed\n");
    free(buffer);
    printf("freed\n");
    return 0;
}
